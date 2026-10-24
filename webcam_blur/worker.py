"""One thread per webcam: feeds its virtual camera, blurring only while an app reads it."""

import sys
import threading
import time

import cv2
import numpy as np

from . import matting, v4l2
from .config import STRENGTHS

IDLE_GRACE = 3.0  # seconds without readers before releasing the real webcam


class CameraError(Exception):
    """Error whose message is a translation key."""


def blur_background(frame, alpha, sigma):
    h, w = frame.shape[:2]
    # Background-weighted blur: the person does not bleed into the blurred background.
    q = cv2.resize(frame, (w // 4, h // 4), interpolation=cv2.INTER_AREA).astype(np.float32)
    bg_w = 1.0 - cv2.resize(alpha, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    num = cv2.GaussianBlur(q * bg_w[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(bg_w, (0, 0), sigma)[..., None] + 1e-4
    bg = cv2.resize(np.clip(num / den, 0, 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.blendLinear(frame, bg, alpha, 1.0 - alpha)


class CameraWorker(threading.Thread):
    def __init__(self, camera, device_nr, config, on_change):
        super().__init__(daemon=True, name=camera.key)
        self.camera = camera
        self.device_nr = device_nr
        self.config = config
        self.settings = config.camera(camera.key)
        self.on_change = on_change
        self.cpu = matting.CpuMatter(matting.CpuMatter.create_session())
        self.in_use = False
        self.engine = ""   # gpu | cpu | gpu_starting | gpu_preparing | gpu_compiling | gpu_error:… | no_gpu | no_onnx
        self.error = None  # translation key or message
        self.stop_evt = threading.Event()

    @property
    def output_path(self):
        return f"/dev/video{self.device_nr}"

    def stop(self, timeout=3):
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

    def capture_mode(self):
        s = self.settings
        if "width" in s and "height" in s:
            return "MJPG", s["width"], s["height"]
        return v4l2.pick_mode(v4l2.capture_modes(self.camera.path))

    def loop(self):
        fmt, w, h = self.capture_mode()
        out = v4l2.LoopbackOutput(self.output_path, w, h)
        if self.error:
            self.error = None
            self.on_change()
        try:
            while not self.stop_evt.is_set():
                if v4l2.readers_of(self.output_path) == 0:
                    out.write_black()
                    self.stop_evt.wait(1.0)
                    continue
                self.stream(out, fmt, w, h)
        finally:
            out.close()

    def gpu_wanted(self):
        """(want GPU, reason if not possible)."""
        state = matting.gpu_support.state
        if not self.config["use_gpu"] or state == "checking":
            return False, "cpu"
        if state != "ok":
            return False, {"no_onnx": "no_onnx"}.get(state, "no_gpu")
        return True, None

    def stream(self, out, fmt, w, h):
        cap = cv2.VideoCapture(self.camera.path, cv2.CAP_V4L2)
        gpu = None
        try:
            if not cap.isOpened():
                raise CameraError("error.open_camera")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fmt))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, self.settings.get("fps", 30))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cpu.reset()
            self.in_use = True
            self.on_change()
            last_reader = time.monotonic()
            next_check = 0.0
            while not self.stop_evt.is_set():
                now = time.monotonic()
                if now >= next_check:
                    next_check = now + 1.0
                    # The GPU subprocess starts in parallel; the CPU covers the first frames.
                    want, reason = self.gpu_wanted()
                    if want and gpu is None:
                        gpu = matting.GpuMatter(w, h, matting.gpu_support.provider)
                    elif not want and gpu is not None:
                        gpu.close()
                        gpu = None
                    if gpu is None:
                        engine = reason
                    elif gpu.ready():
                        engine = "gpu"
                    else:
                        engine = "gpu_" + gpu.status
                    if engine != self.engine:
                        self.engine = engine
                        self.on_change()
                    if v4l2.readers_of(self.output_path) > 0:
                        last_reader = now
                    elif now - last_reader > IDLE_GRACE:
                        return
                ok, frame = cap.read()
                if not ok:
                    raise CameraError("error.read_camera")
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, (w, h))
                if self.settings["blur"]:
                    a = gpu.alpha(frame) if gpu is not None and gpu.ready() else None
                    if a is None:
                        a = self.cpu.alpha(frame)
                    frame = blur_background(frame, a, STRENGTHS[self.config["strength"]])
                out.write(frame)
        finally:
            cap.release()
            if gpu is not None:
                gpu.close()
            self.in_use = False
            self.engine = ""
            self.on_change()
