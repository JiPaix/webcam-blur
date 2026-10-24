"""Whether the GPU path can be offered: checked once at startup, in a throwaway process.

The check (``python -m webcam_blur.gpucheck``) runs the "low" model on a portrait with
ncnn on Vulkan and on the CPU: the GPU must give the same cut-out (drivers can have bugs:
see tools/convert_ncnn.py) and be faster (a software Vulkan driver is not). A broken
driver cannot crash the app, and nothing stays on the GPU.
"""

import json
import subprocess
import sys
import threading
from pathlib import Path

from .proc import module_command

PROBE = Path(__file__).resolve().parent / "data" / "probe.jpg"


class GpuSupport:
    """``state``: checking | ok | no_ncnn (ncnn not installed) | unsupported"""

    def __init__(self):
        self.state = "checking"
        self.detail = ""

    def check(self, on_done=None, timeout=90):
        def run():
            cmd, env = module_command("gpucheck")
            try:
                out = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
                result = json.loads(out.stdout.strip().splitlines()[-1])
            except subprocess.TimeoutExpired:
                result = {"state": "unsupported", "detail": "GPU check timed out"}
            except (ValueError, IndexError):
                result = {"state": "unsupported", "detail": f"GPU check failed (exit code {out.returncode})"}
            self.detail = result["detail"]
            self.state = result["state"]
            if self.state != "ok":
                print(f"GPU acceleration unavailable ({self.detail})", file=sys.stderr, flush=True)
            if on_done:
                on_done()
        threading.Thread(target=run, daemon=True, name="gpu-check").start()


gpu_support = GpuSupport()


def detect_gpu(runs=5):
    """{"state": ok | no_ncnn | unsupported, "detail"}"""
    import time

    import cv2
    import numpy as np

    from . import ncnn
    from .models import QUALITY_MODELS

    if ncnn.library() is None:
        return {"state": "no_ncnn", "detail": "ncnn is not installed"}
    if ncnn.gpu_count() == 0:
        return {"state": "unsupported", "detail": "no Vulkan GPU found by ncnn"}
    frame = cv2.imread(str(PROBE))
    h, w = frame.shape[:2]
    results = {}
    try:
        for vulkan in (False, True):
            net = ncnn.RvmNet(QUALITY_MODELS["low"], w, h, vulkan=vulkan)
            try:
                net.alpha(frame)  # first run: shader compilation, buffers
                start = time.perf_counter()
                for _ in range(runs):
                    alpha = net.alpha(frame)
                results[vulkan] = (alpha, (time.perf_counter() - start) / runs)
            finally:
                net.close()
    except Exception as e:
        return {"state": "unsupported", "detail": f"{type(e).__name__}: {e}"}
    (gpu, gpu_time), (cpu, cpu_time) = results[True], results[False]
    diff = float(np.abs(gpu - cpu).mean())
    if diff > 0.005:
        return {"state": "unsupported", "detail": f"the GPU's results differ from the CPU's ({diff:.3f})"}
    if gpu_time >= cpu_time:
        return {"state": "unsupported",
                "detail": f"the GPU is not faster than the CPU ({gpu_time * 1000:.1f} vs {cpu_time * 1000:.1f} ms)"}
    return {"state": "ok", "detail": f"{gpu_time * 1000:.1f} ms per frame (CPU: {cpu_time * 1000:.1f} ms)"}


def main():
    print(json.dumps(detect_gpu()), flush=True)


if __name__ == "__main__":
    main()
