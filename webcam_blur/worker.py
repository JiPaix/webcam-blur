"""One supervisor thread per webcam, in the tray process.

It keeps the virtual camera ready (format set, black frames while idle) and, while an app
reads it, runs a stream subprocess (stream.py) that does the actual work. The tray process
itself never loads OpenCV, onnxruntime or a model.
"""

import json
import subprocess
import sys
import threading
import time
from dataclasses import asdict

from . import gpucheck, v4l2
from .config import STRENGTHS
from .models import GPU_MODELS
from .proc import module_command

IDLE_GRACE = 3.0  # seconds without readers before stopping the stream (and releasing the webcam)


class CameraError(Exception):
    """Error whose message is a translation key."""


class CameraWorker(threading.Thread):
    def __init__(self, camera, device_nr, config, on_change):
        super().__init__(daemon=True, name=camera.key)
        self.camera = camera
        self.device_nr = device_nr
        self.config = config
        self.settings = config.camera(camera.key)
        self.on_change = on_change
        self.modes = []    # every mode the webcam offers (v4l2.Mode), best first
        self.mode = None   # mode the virtual camera currently has
        self.proc = None   # stream subprocess, while an app reads the virtual camera
        self.in_use = False
        self.engine = ""   # gpu | cpu | gpu_starting | gpu_preparing | gpu_compiling | gpu_error:… | no_gpu | no_onnx
        self.error = None  # translation key or message
        self.stop_evt = threading.Event()

    @property
    def output_path(self):
        return f"/dev/video{self.device_nr}"

    def stop(self, timeout=8):
        self.stop_evt.set()
        self.join(timeout)

    def run(self):
        while not self.stop_evt.is_set():
            try:
                self.loop()
            except Exception as e:  # webcam unplugged or busy…
                self.error = str(e)
                self.in_use = False
                self.on_change()
                if not isinstance(e, CameraError):
                    print(f"[{self.camera.name}] {e}", file=sys.stderr, flush=True)
                self.stop_evt.wait(5)

    def wanted_mode(self):
        """The mode chosen in the menu if the webcam offers it, else the automatic one."""
        return v4l2.find_mode(self.modes, self.settings.get("mode")) or v4l2.auto_mode(self.modes)

    @property
    def mode_pending(self):
        """A new mode was chosen while an app reads the camera: it applies once the app is gone."""
        return self.mode is not None and self.mode != self.wanted_mode()

    def loop(self):
        if not self.modes:
            self.modes = v4l2.list_modes(self.camera.path)
            self.on_change()
        mode = self.wanted_mode()
        # The format can only change while no app reads the virtual camera.
        out = v4l2.LoopbackOutput(self.output_path, mode.width, mode.height)
        out.set_fps(mode.fps)
        self.mode = mode
        if self.error:
            self.error = None
        self.on_change()
        try:
            while not self.stop_evt.is_set() and self.wanted_mode() == mode:
                if v4l2.readers_of(self.output_path) == 0:
                    out.write_black()
                    # Readers can overwrite the advertised rate (it is shared, and e.g. OpenCV
                    # requests 30 fps on open): restore it for the next app.
                    out.set_fps(mode.fps)
                    self.stop_evt.wait(1.0)
                    continue
                self.stream(out, mode)
        finally:
            out.close()

    def stream_settings(self):
        """What the stream subprocess needs to know; sent again whenever it changes."""
        support = gpucheck.gpu_support
        use_gpu = self.config["use_gpu"] and support.state == "ok"
        if use_gpu or not self.config["use_gpu"] or support.state == "checking":
            reason = "cpu"
        else:
            reason = "no_onnx" if support.state == "no_onnx" else "no_gpu"
        return {"blur": self.settings["blur"], "sigma": STRENGTHS[self.config["strength"]],
                "gpu_provider": support.provider if use_gpu else None,
                "gpu_model": GPU_MODELS[self.config["quality"]], "cpu_reason": reason}

    def stream(self, out, mode):
        cmd, env = module_command("stream")
        proc = subprocess.Popen(cmd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                text=True, pass_fds=(out.fd,))
        self.proc = proc
        errors = []

        def read_status():
            for line in proc.stdout:
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if "engine" in message:
                    self.engine = message["engine"]
                    self.on_change()
                if "error" in message:
                    errors.append(message["error"])

        reader = threading.Thread(target=read_status, daemon=True)
        reader.start()
        sent = self.stream_settings()
        try:
            self._send(proc, {"camera": self.camera.path, "fd": out.fd, "mode": asdict(mode), "settings": sent})
            self.in_use = True
            self.on_change()
            last_reader = time.monotonic()
            while not self.stop_evt.wait(1.0):
                if proc.poll() is not None:
                    reader.join(1)
                    if errors:
                        raise CameraError(errors[0])
                    raise RuntimeError(f"stream process exited with code {proc.returncode}")
                current = self.stream_settings()
                if current != sent:
                    self._send(proc, current)
                    sent = current
                now = time.monotonic()
                if v4l2.readers_of(self.output_path, exclude=(proc.pid,)) > 0:
                    last_reader = now
                elif now - last_reader > IDLE_GRACE:
                    return
        finally:
            self._send(proc, {"stop": True})
            try:
                proc.stdin.close()
            except OSError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            reader.join(1)
            self.proc = None
            self.in_use = False
            self.engine = ""
            self.on_change()

    @staticmethod
    def _send(proc, message):
        try:
            proc.stdin.write(json.dumps(message) + "\n")
            proc.stdin.flush()
        except OSError:  # the subprocess already exited
            pass
