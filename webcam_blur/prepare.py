"""Preparing GPU models ahead of time: freezing for one input size and, on MIGraphX, compiling.

Compiling takes 1-2 minutes of every core per size, so it runs in its own process at the
lowest CPU priority, and never competes with a call. The tray starts it for each webcam's
resolution in every quality (see Precompiler), so the GPU is ready when an app opens a
camera; a stream that needs a size not prepared yet runs it too (see matting.gpu_process).
A lock per size keeps two of them from compiling the same thing.

usage: python -m webcam_blur.prepare, with [[model, width, height, provider], ...] as JSON on stdin

The tray imports this module: heavy imports stay inside the functions run by the subprocess.
"""

import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

from .models import CACHE, GPU_MODELS, gpu_input_size
from .proc import module_command

ROCM_VERSION = Path("/opt/rocm/.info/version")


def stamp():
    """What a compiled model depends on, besides the model and its size."""
    import onnxruntime as ort
    try:
        rocm = ROCM_VERSION.read_text().strip()
    except OSError:
        rocm = "unknown"
    return f"onnxruntime {ort.__version__}, ROCm {rocm}\n"


def is_prepared(model, width, height, provider):
    from .matting import frozen_path, migraphx_cache
    path = frozen_path(model, width, height)
    if not path.is_file():
        return False
    if provider != "MIGraphXExecutionProvider":
        return True  # other providers compile nothing ahead
    try:
        return (migraphx_cache(path) / "stamp").read_text() == stamp()
    except OSError:
        return False


def prepare(model, width, height, provider):
    from .matting import _gpu_session, frozen_model, migraphx_cache
    lock_path = CACHE / "models" / f"{model.removesuffix('_fp32.onnx')}_{width}x{height}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # another process may be preparing this size
        if is_prepared(model, width, height, provider):
            return
        path = frozen_model(model, width, height)
        if provider == "MIGraphXExecutionProvider":
            _gpu_session(path, provider)  # compiles into the cache
            (migraphx_cache(path) / "stamp").write_text(stamp())


def jobs_for(modes, quality, provider):
    """Sizes to prepare for these webcam modes: the current quality first, then the others."""
    jobs = []
    for q in [quality] + [q for q in GPU_MODELS if q != quality]:
        for mode in modes:
            job = [GPU_MODELS[q], *gpu_input_size(mode.width, mode.height, q), provider]
            if job not in jobs:
                jobs.append(job)
    return jobs


class Precompiler:
    """Tray side: keeps a preparation process running until the wanted sizes are ready."""

    def __init__(self):
        self.proc = None
        self.done = None  # the jobs last run to the end (successfully or not)
        self.running = None

    def update(self, jobs):
        if self.proc is not None and self.proc.poll() is not None:
            if self.proc.returncode:
                print(f"model preparation failed (exit code {self.proc.returncode})",
                      file=sys.stderr, flush=True)
            self.proc, self.done = None, self.running
        wanted = {tuple(job) for job in jobs}  # the order only sets priorities
        if self.proc is None and wanted and wanted != self.done:
            cmd, env = module_command("prepare")
            self.proc = subprocess.Popen(cmd, env=env, stdin=subprocess.PIPE, text=True)
            self.proc.stdin.write(json.dumps(jobs))
            self.proc.stdin.close()
            self.running = wanted

    def stop(self):
        # A compile left running finishes on its own (at the lowest priority) rather than
        # being killed halfway through writing the cache.
        self.proc = None


def run(jobs):
    """Prepare each job; returns the number of failures."""
    failures = 0
    for model, width, height, provider in jobs:
        try:
            prepare(model, width, height, provider)
        except Exception as e:
            failures += 1
            print(f"cannot prepare {model} at {width}x{height}: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
    return failures


def main():
    os.nice(19)
    sys.exit(1 if run(json.loads(sys.stdin.read())) else 0)


if __name__ == "__main__":
    main()
