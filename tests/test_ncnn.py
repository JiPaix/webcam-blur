"""The GPU path: RVM on ncnn. Runs on ncnn's CPU backend where there is no GPU (CI), which
uses the same converted models and code; the Vulkan comparison needs a GPU."""

import os
import time

import cv2
import numpy as np
import onnxruntime as ort
import pytest
from conftest import BACKGROUND, PERSON, region

from webcam_blur import gpucheck, ncnn
from webcam_blur.models import QUALITY_MODELS, model_path

pytestmark = [pytest.mark.usefixtures("models"),
              pytest.mark.skipif(ncnn.library() is None, reason="ncnn is not installed")]

MBV3, R50 = QUALITY_MODELS["standard"], QUALITY_MODELS["high"]
needs_gpu = pytest.mark.skipif(ncnn.library() is None or ncnn.gpu_count() == 0, reason="no Vulkan GPU")


def onnx_alpha(model, frame, steps=5):
    """Reference: the original ONNX model on onnxruntime, at the frame's size."""
    h, w = frame.shape[:2]
    sess = ort.InferenceSession(str(model_path(model)), providers=["CPUExecutionProvider"])
    x = (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255).transpose(2, 0, 1)[None].copy()
    rec = [np.zeros((1, 1, 1, 1), np.float32)] * 4
    for _ in range(steps):
        _, pha, *rec = sess.run(None, {"src": x, **{f"r{i}i": rec[i - 1] for i in range(1, 5)},
                                       "downsample_ratio": np.array([min(1.0, 320 / max(w, h))], np.float32)})
    return pha[0, 0], [r.shape[1:] for r in rec]


def ncnn_alpha(model, frame, steps=5, vulkan=False):
    h, w = frame.shape[:2]
    net = ncnn.RvmNet(model, w, h, vulkan=vulkan)
    try:
        for _ in range(steps):
            a = net.alpha(frame)
    finally:
        net.close()
    return a


def assert_good_matte(a, h, w):
    """PERSON/BACKGROUND regions are given at 1280x720."""
    a = cv2.resize(a, (1280, 720))
    for name in PERSON:
        assert region(a, name).mean() > 0.95, f"{name} should be foreground"
    for name in BACKGROUND:
        assert region(a, name).mean() < 0.05, f"{name} should be background"


@pytest.mark.parametrize("model", [MBV3, R50])
@pytest.mark.parametrize("size", [(640, 360), (640, 480), (1280, 720)])
def test_matches_onnxruntime_at_any_size(frame, model, size):
    """The 640x360 template, resized for each frame size, computes what the ONNX model does."""
    small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    ref, _ = onnx_alpha(model, small)
    a = ncnn_alpha(model, small)
    assert a.shape == ref.shape
    assert ((a > 0.5) == (ref > 0.5)).mean() > 0.999
    assert np.abs(a - ref).mean() < 0.002


@pytest.mark.parametrize("model", [MBV3, R50])
def test_state_shapes_follow_the_size(model):
    template, _ = ncnn.model_files(model)
    for w, h in ((640, 360), (640, 480), (1920, 1080), (320, 240)):
        _, states = ncnn.sized_param(template.read_text(), w, h)
        _, expected = onnx_alpha(model, np.zeros((h, w, 3), np.uint8), steps=1)
        assert states == [tuple(s) for s in expected], f"{w}x{h}"


def test_template_without_input_shapes_is_refused():
    template, _ = ncnn.model_files(MBV3)
    import re
    bare = re.sub(r" 0=\d+ 1=\d+ 2=\d+", "", template.read_text())
    with pytest.raises(RuntimeError, match="input shapes"):
        ncnn.sized_param(bare, 640, 360)


def test_gpu_matter_loads_in_the_background(frame):
    gpu = ncnn.GpuMatter(640, 360, MBV3, vulkan=False)
    try:
        deadline = time.monotonic() + 60
        while not gpu.ready():
            assert not gpu.status.startswith("error"), gpu.status
            assert time.monotonic() < deadline, "model did not load"
            time.sleep(0.05)
        for _ in range(5):
            a = gpu.alpha(frame)
        assert a.shape == frame.shape[:2], "cut-out scaled back to the frame's size"
        assert_good_matte(a, *frame.shape[:2])
    finally:
        gpu.close()
    assert gpu.net is None


def test_gpu_matter_reports_load_errors():
    gpu = ncnn.GpuMatter(640, 360, "no_such_model_fp32.onnx", vulkan=False)
    gpu.thread.join(30)
    assert gpu.status.startswith("error:")
    assert not gpu.ready()
    gpu.close()


@needs_gpu
@pytest.mark.parametrize("model", [MBV3, R50])
def test_vulkan_matches_cpu(frame, model):
    small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
    gpu, cpu = ncnn_alpha(model, small, vulkan=True), ncnn_alpha(model, small)
    assert np.abs(gpu - cpu).mean() < 0.002


def test_gpu_check_without_ncnn(monkeypatch):
    monkeypatch.setattr(ncnn, "library", lambda: None)
    assert gpucheck.detect_gpu()["state"] == "no_ncnn"


def test_gpu_check_without_a_gpu(monkeypatch):
    monkeypatch.setattr(ncnn, "gpu_count", lambda: 0)
    assert gpucheck.detect_gpu()["state"] == "unsupported"


def test_gpu_check_refuses_wrong_results(monkeypatch):
    """A driver bug (like ncnn's Vulkan Reduction on RDNA 4) must not reach the user."""
    monkeypatch.setattr(ncnn, "gpu_count", lambda: 1)
    real = ncnn.RvmNet

    class Buggy(real):
        def alpha(self, frame):
            a = real.alpha(self, frame)
            return a if self.vulkan_requested is False else 1.0 - a

    def make(model, w, h, vulkan=True, threads=2):
        net = Buggy(model, w, h, vulkan=False)
        net.vulkan_requested = vulkan
        return net

    monkeypatch.setattr(ncnn, "RvmNet", make)
    result = gpucheck.detect_gpu()
    assert result["state"] == "unsupported" and "differ" in result["detail"]


@needs_gpu
@pytest.mark.skipif(bool(os.environ.get("WEBCAM_BLUR_SOFTWARE_VULKAN")), reason="software Vulkan is slow")
def test_gpu_check_on_this_machine():
    result = gpucheck.detect_gpu()
    assert result["state"] == "ok", result["detail"]
