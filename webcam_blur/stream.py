"""Stream subprocess: webcam -> person matting -> blurred background -> virtual camera.

The tray starts it (see worker.py) when an app opens a virtual camera and stops it when
the app is gone, so OpenCV, onnxruntime and the models are only in memory while a camera
is in use. It writes frames through the virtual camera's file descriptor, which the tray
opened and configured and hands over (the format cannot change under a reading app).

Protocol, as JSON lines:
- stdin, first line: {"camera": path, "fd": n, "mode": {width, height, fps, fourcc}, "settings": {…}}
- stdin, then: updated settings {"blur", "sigma", "gpu_provider", "cpu_reason"}, or {"stop": true}
- stdout: {"engine": …} whenever it changes, {"error": <translation key>} before failing
"""

import json
import sys
import threading
import time

import cv2
import numpy as np

from . import matting, v4l2


class StreamError(Exception):
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


def send(**message):
    print(json.dumps(message), flush=True)


def run(camera, out, mode, settings, stop):
    w, h = mode.width, mode.height
    cap = cv2.VideoCapture(camera, cv2.CAP_V4L2)
    gpu = None
    try:
        if not cap.isOpened():
            raise StreamError("error.open_camera")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*mode.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, mode.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cpu = matting.CpuMatter(matting.CpuMatter.create_session())
        engine = None
        next_check = 0.0
        while not stop.is_set():
            now = time.monotonic()
            if now >= next_check:
                next_check = now + 1.0
                # The GPU subprocess starts in parallel; the CPU covers the first frames.
                provider = settings.get("gpu_provider")
                if provider and gpu is None:
                    gpu = matting.GpuMatter(w, h, provider)
                elif not provider and gpu is not None:
                    gpu.close()
                    gpu = None
                if gpu is None:
                    current = settings.get("cpu_reason", "cpu")
                else:
                    current = "gpu" if gpu.ready() else "gpu_" + gpu.status
                if current != engine:
                    engine = current
                    send(engine=engine)
            ok, frame = cap.read()
            if not ok:
                raise StreamError("error.read_camera")
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h))
            if settings["blur"]:
                a = gpu.alpha(frame) if gpu is not None and gpu.ready() else None
                if a is None:
                    a = cpu.alpha(frame)
                frame = blur_background(frame, a, settings["sigma"])
            out.write(frame)
    finally:
        cap.release()
        if gpu is not None:
            gpu.close()


def main():
    init = json.loads(sys.stdin.readline())
    settings = dict(init["settings"])
    stop = threading.Event()

    def read_commands():
        for line in sys.stdin:
            message = json.loads(line)
            if message.get("stop"):
                break
            settings.update(message)
        stop.set()  # asked to stop, or the tray went away (end of stdin)

    threading.Thread(target=read_commands, daemon=True).start()
    mode = v4l2.Mode(**init["mode"])
    out = v4l2.LoopbackOutput.attach(init["fd"], mode.width, mode.height)
    try:
        run(init["camera"], out, mode, settings, stop)
    except StreamError as e:
        send(error=str(e))
        sys.exit(2)
    finally:
        out.close()


if __name__ == "__main__":
    main()
