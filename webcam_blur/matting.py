"""Person matting with Robust Video Matting (RVM) on the CPU, with onnxruntime.

RVM mobilenetv3 on frames downscaled to at most 320 px ("low" quality) or 640 px wide
("standard"), resnet50 at 640 px ("high"). The GPU path is in ncnn.py (Vulkan).
"""

import ctypes
import os

import cv2
import numpy as np
import onnxruntime as ort

from .models import CPU_MODEL, QUALITY_MODELS, gpu_input_size, model_path  # noqa: F401  (re-exported)

# Widest frame the model sees on the CPU, per quality (GPU: models.GPU_WIDTH). The cut-out
# is scaled back up: edges barely change, and a smaller input is much cheaper.
CPU_WIDTH = {"low": 320, "standard": 640, "high": 640}

# OpenCV's default (one thread per core) spends more time handing out work and waiting
# than computing on frames this size.
CV_THREADS = 2


def session_options(threads):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    # Idle threads spin waiting for work by default, burning CPU between frames.
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return opts


def release_memory():
    """Give memory freed by model loading back to the system (glibc keeps it otherwise)."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass  # not glibc


def downsample_ratio(width, height, target=320):
    """RVM runs its recurrent part at about ``target`` px on the long side."""
    return min(1.0, target / max(width, height))


class CpuMatter:
    """RVM on CPU; the recurrent state belongs to one camera."""

    def __init__(self, session, width=CPU_WIDTH["standard"], model=CPU_MODEL):
        self.session, self.width, self.model = session, width, model
        self.reset()

    @staticmethod
    def create_session(threads=None, model=CPU_MODEL):
        # Without spinning, extra threads cost little and keep 60 fps within each frame's time.
        threads = threads or max(1, min(6, (os.cpu_count() or 2) // 2))
        return ort.InferenceSession(str(model_path(model)), session_options(threads),
                                    providers=["CPUExecutionProvider"])

    def set_width(self, width):
        if width != self.width:
            self.width = width
            self.reset()  # the recurrent state has the old size

    def reset(self):
        self.rec = [np.zeros((1, 1, 1, 1), np.float32)] * 4

    def alpha(self, frame):
        h, w = frame.shape[:2]
        scale = min(1.0, self.width / w)
        small = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) \
            if scale < 1 else frame
        sh, sw = small.shape[:2]
        x = (cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        _fgr, pha, *self.rec = self.session.run(None, {
            "src": x, "r1i": self.rec[0], "r2i": self.rec[1], "r3i": self.rec[2], "r4i": self.rec[3],
            "downsample_ratio": np.array([downsample_ratio(sw, sh, self.width // 2)], np.float32),
        })
        a = pha[0, 0]
        return cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR) if a.shape != (h, w) else a
