"""Stream subprocess: webcam -> person matting -> blurred background -> virtual camera.

The tray starts it (see worker.py) when an app opens a virtual camera and stops it when
the app is gone, so OpenCV, onnxruntime and the models are only in memory while a camera
is in use. It writes frames through the virtual camera's file descriptor, which the tray
opened and configured and hands over (the format cannot change under a reading app).

Protocol, as JSON lines:
- stdin, first line: {"camera": path, "fd": n, "mode": {width, height, fps, fourcc}, "settings": {…}}
- stdin, then: updated settings {"blur", "sigma", "quality", "gpu" (bool), "cpu_reason"},
  or {"stop": true}
- stdout: {"engine": …} whenever it changes, {"error": <translation key>} before failing
"""

import gc
import json
import sys
import threading
import time

import cv2
import numpy as np

from . import matting, ncnn, v4l2
from .models import QUALITY_MODELS


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


MAX_INTERVAL = 6


def matting_interval(fps, quality):
    """Run the model on every n-th frame: "low" updates the cut-out at least about 30 times a
    second, reusing it in between; the others run it on every frame. ``fps`` is the rate the
    webcam really delivers: many lower it in dim light (e.g. 60 -> 30)."""
    return max(1, int(fps * 1.1 / 30)) if quality == "low" else 1


def adapt_interval(n, base, load, model_time, rest_time, fps):
    """Run the model less often when it cannot keep up (e.g. "high" on a slow CPU): the video
    keeps the webcam's frame rate and the cut-out updates less often, instead of frames being
    dropped. ``load``: share of the last second spent processing; times in seconds."""
    n = max(n, base)
    if load > 0.95:  # no time left between frames: falling behind (two capture buffers absorb less)
        return min(n + 1, MAX_INTERVAL)
    if n > base and (model_time / (n - 1) + rest_time) * fps < 0.92:
        return n - 1  # room again, e.g. after a busy moment or switching to a lighter quality
    return n


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
        # Two buffers: a frame where the model runs longer than a frame lasts (e.g. "high" on the
        # CPU) does not lose the next one; frames are still at most ~20 ms old when processed.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        cpu = None
        engine = None
        next_check = 0.0
        a, frame_no, interval = None, 0, 1
        frames, counted_since, fps, busy = 0, time.monotonic(), mode.fps, 0.0
        model_time = rest_time = 0.0  # moving averages, seconds per frame
        while not stop.is_set():
            now = time.monotonic()
            if now >= next_check:
                next_check = now + 1.0
                quality, use_gpu = settings.get("quality", "low"), settings.get("gpu", False)
                if frames >= 10:
                    elapsed = now - counted_since
                    fps, load = frames / elapsed, busy / elapsed
                    frames, counted_since, busy = 0, now, 0.0
                    interval = adapt_interval(interval, matting_interval(fps, quality), load,
                                              model_time, rest_time, fps)
                interval = max(interval, matting_interval(fps, quality))
                # The GPU model loads in a thread; the CPU covers the first frames.
                model = QUALITY_MODELS[quality]
                size = matting.gpu_input_size(w, h, quality)
                if gpu is not None and (not use_gpu or (gpu.model, gpu.width, gpu.height) != (model, *size)):
                    gpu.close()  # GPU turned off, or quality changed
                    gpu = None
                if use_gpu and gpu is None:
                    gpu = ncnn.GpuMatter(*size, model)
                if gpu is None:
                    current = settings.get("cpu_reason", "cpu")
                else:
                    current = "gpu" if gpu.ready() else "gpu_" + gpu.status
                if current != engine:
                    engine = current
                    send(engine=engine)
                if current == "gpu" and cpu is not None:
                    cpu = None  # the GPU took over: free the CPU model (~80 MB)
                    gc.collect()
                    matting.release_memory()
            if not cap.grab():  # waits for the next frame
                raise StreamError("error.read_camera")
            started, matted = time.monotonic(), 0.0  # busy from here: decoding, matting, blur
            ok, frame = cap.retrieve()
            if not ok:
                raise StreamError("error.read_camera")
            frames += 1
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h))
            if settings["blur"]:
                if a is None or frame_no % interval == 0:
                    matting_started = time.monotonic()
                    a = gpu.alpha(frame) if gpu is not None and gpu.ready() else None
                    if a is None:
                        model = QUALITY_MODELS[quality]
                        if cpu is None or cpu.model != model:  # first frames, GPU failed, quality changed
                            loading = time.monotonic()
                            cpu = matting.CpuMatter(matting.CpuMatter.create_session(model=model),
                                                    model=model)
                            loaded = time.monotonic() - loading  # not per-frame work: left out of the load
                            started += loaded
                            matting_started += loaded
                        cpu.set_width(matting.CPU_WIDTH[quality])
                        a = cpu.alpha(frame)
                    matted = time.monotonic() - matting_started
                    model_time = 0.8 * model_time + 0.2 * matted
                frame_no += 1
                frame = blur_background(frame, a, settings["sigma"])
            out.write(frame)
            spent = time.monotonic() - started
            busy += spent
            rest_time = 0.8 * rest_time + 0.2 * (spent - matted)
    finally:
        cap.release()
        if gpu is not None:
            gpu.close()


def main():
    cv2.setNumThreads(matting.CV_THREADS)
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
