"""Startup check: does GPU inference work on this machine?

The check runs ``python -m webcam_blur.gpucheck`` in a throwaway process, so the tray
process never loads onnxruntime, a broken driver cannot crash or hang it, and nothing
stays allocated on the GPU afterwards.
"""

import json
import subprocess
import sys
import threading

from .proc import module_command


class GpuSupport:
    """``state``: checking | ok | no_onnx (GPU works, python-onnx missing) | unsupported"""

    def __init__(self):
        self.state = "checking"
        self.provider = None
        self.detail = ""

    def check(self, on_done=None, timeout=90):
        def run():
            cmd, env = module_command("gpucheck")
            try:
                out = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
                result = json.loads(out.stdout.strip().splitlines()[-1])
            except subprocess.TimeoutExpired:
                result = {"state": "unsupported", "provider": None, "detail": "GPU check timed out"}
            except (ValueError, IndexError):
                result = {"state": "unsupported", "provider": None,
                          "detail": f"GPU check failed (exit code {out.returncode})"}
            self.provider, self.detail = result["provider"], result["detail"]
            self.state = result["state"]
            if self.state == "unsupported":
                print(f"GPU acceleration unavailable ({self.detail})", file=sys.stderr, flush=True)
            if on_done:
                on_done()
        threading.Thread(target=run, daemon=True, name="gpu-check").start()


gpu_support = GpuSupport()


def main():
    from .matting import detect_gpu
    print(json.dumps(detect_gpu()), flush=True)


if __name__ == "__main__":
    main()
