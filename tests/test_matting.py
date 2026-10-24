"""Person matting and blur on the public-domain test portrait (CPU only, no device needed).

The "GPU" pipeline is exercised with onnxruntime's CPU provider: model freezing, the
subprocess, shared memory and its clean shutdown are the same as on a real GPU.
"""

import os
import time

import numpy as np
import onnxruntime as ort
import pytest
from conftest import BACKGROUND, PERSON, TEXTURED, detail, mean_abs_diff, region

from webcam_blur import matting
from webcam_blur.stream import blur_background, matting_interval

pytestmark = pytest.mark.usefixtures("models")


def warm_alpha(matter, frame, n=8):
    """RVM is recurrent: let it settle on the frame first."""
    for _ in range(n):
        a = matter.alpha(frame)
    return a


def assert_good_matte(a):
    assert a.dtype == np.float32
    for name in PERSON:
        assert region(a, name).mean() > 0.95, f"{name} should be foreground"
    for name in BACKGROUND:
        assert region(a, name).mean() < 0.05, f"{name} should be background"


@pytest.mark.parametrize("quality", ["low", "standard"])
def test_cpu_matting(frame, quality):
    matter = matting.CpuMatter(matting.CpuMatter.create_session(), matting.CPU_WIDTH[quality])
    a = warm_alpha(matter, frame)
    assert a.shape == frame.shape[:2]
    assert_good_matte(a)


def test_low_quality_cut_out_is_close_to_standard(frame):
    session = matting.CpuMatter.create_session()
    low = warm_alpha(matting.CpuMatter(session, matting.CPU_WIDTH["low"]), frame)
    standard = warm_alpha(matting.CpuMatter(session, matting.CPU_WIDTH["standard"]), frame)
    assert np.abs(low - standard).mean() < 0.02


def test_cpu_matter_width_change_resets_its_state(frame):
    matter = matting.CpuMatter(matting.CpuMatter.create_session(), matting.CPU_WIDTH["standard"])
    warm_alpha(matter, frame, 2)
    matter.set_width(matting.CPU_WIDTH["low"])
    assert_good_matte(warm_alpha(matter, frame))  # would fail on a recurrent state of the old size


@pytest.mark.parametrize("size, quality, expected", [
    ((1280, 720), "low", (640, 360)),
    ((1920, 1080), "low", (640, 360)),
    ((640, 480), "low", (640, 480)),
    ((1280, 720), "standard", (1280, 720)),
    ((1920, 1080), "high", (1920, 1080)),
])
def test_gpu_input_size(size, quality, expected):
    assert matting.gpu_input_size(*size, quality) == expected


@pytest.mark.parametrize("fps, quality, expected", [
    (30, "low", 1), (60, "low", 2), (50, "low", 1), (15, "low", 1), (60, "standard", 1), (60, "high", 1),
])
def test_matting_interval(fps, quality, expected):
    assert matting_interval(fps, quality) == expected


def test_cpu_matting_other_resolution(frame):
    import cv2
    small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
    a = warm_alpha(matting.CpuMatter(matting.CpuMatter.create_session()), small)
    assert a.shape == (360, 640)
    assert a[85:160, 290:360].mean() > 0.95  # face
    assert a[:, :140].mean() < 0.05          # left checkerboard


def test_blur_background(frame):
    matter = matting.CpuMatter(matting.CpuMatter.create_session())
    a = warm_alpha(matter, frame)
    out = blur_background(frame, a, 6.0)
    for name in TEXTURED:
        assert detail(out, name) < 0.3 * detail(frame, name), f"{name} is not blurred"
    for name in PERSON:
        assert mean_abs_diff(out, frame, name) < 2.0, f"{name} was altered"


def test_blur_strengths_are_ordered(frame):
    a = warm_alpha(matting.CpuMatter(matting.CpuMatter.create_session()), frame)
    values = [detail(blur_background(frame, a, s), "flag") for s in (3.0, 6.0, 12.0)]
    assert values[0] > values[1] > values[2]


@pytest.mark.parametrize("quality", ["standard", "high"])
def test_frozen_model_matches_original(frame, quality):
    """The static-shape model built for GPU providers computes the same thing."""
    h, w = frame.shape[:2]
    model = matting.GPU_MODELS[quality]
    path = matting.frozen_model(model, w, h)
    frozen = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    original = ort.InferenceSession(str(matting.model_path(model)), providers=["CPUExecutionProvider"])
    shapes = {i.name: i.shape for i in frozen.get_inputs()}
    assert shapes["src"] == [1, 3, h, w]
    assert "downsample_ratio" not in shapes
    x = (frame[..., ::-1].astype(np.float32) / 255).transpose(2, 0, 1)[None].copy()
    rec = [np.zeros(shapes[f"r{k}i"], np.float32) for k in range(1, 5)]
    rec_o = [np.zeros((1, 1, 1, 1), np.float32)] * 4
    for _ in range(3):
        _, pha_f, *rec = frozen.run(None, {"src": x, **{f"r{k}i": rec[k - 1] for k in range(1, 5)}})
        _, pha_o, *rec_o = original.run(None, {
            "src": x, **{f"r{k}i": rec_o[k - 1] for k in range(1, 5)},
            "downsample_ratio": np.array([matting.downsample_ratio(w, h)], np.float32)})
    assert np.abs(pha_f - pha_o).max() < 1e-3
    assert_good_matte(pha_f[0, 0])


def shm_entries():
    return {f for f in os.listdir("/dev/shm") if f.startswith("psm_")}


@pytest.mark.parametrize("quality", ["low", "standard", "high"])
def test_gpu_subprocess_pipeline(frame, quality):
    h, w = frame.shape[:2]
    before = shm_entries()
    size = matting.gpu_input_size(w, h, quality)
    gpu = matting.GpuMatter(*size, "CPUExecutionProvider", matting.GPU_MODELS[quality])
    try:
        deadline = time.monotonic() + 180
        while not gpu.ready():
            assert not gpu.status.startswith("error"), gpu.status
            assert time.monotonic() < deadline, f"stuck in {gpu.status}"
            time.sleep(0.2)
        assert gpu.proc.is_alive()
        for _ in range(4):
            a = gpu.alpha(frame)
            assert a is not None, gpu.status
        assert a.shape == (h, w)
        assert_good_matte(a)
    finally:
        gpu.close()
    assert not gpu.proc.is_alive(), "GPU subprocess must exit with the stream"
    assert shm_entries() == before, "shared memory leaked"


def test_gpu_subprocess_reports_errors(frame):
    """A provider that cannot load must surface as an error (the worker then stays on CPU)."""
    h, w = frame.shape[:2]
    gpu = matting.GpuMatter(w, h, "NoSuchExecutionProvider")
    try:
        deadline = time.monotonic() + 120
        while not gpu.status.startswith("error"):
            assert not gpu.ready(), "an unknown provider cannot become ready"
            assert time.monotonic() < deadline, f"no error reported ({gpu.status})"
            time.sleep(0.2)
    finally:
        gpu.close()
    assert not gpu.proc.is_alive()


def test_try_provider():
    assert matting.try_provider("CPUExecutionProvider") is None
    assert matting.try_provider("NoSuchExecutionProvider")


@pytest.mark.parametrize("provider, onnx_installed, expected", [
    (None, True, "unsupported"),                         # CPU-only onnxruntime
    ("NoSuchExecutionProvider", True, "unsupported"),    # provider listed but not working
    ("CPUExecutionProvider", False, "no_onnx"),          # works, python-onnx missing
    ("CPUExecutionProvider", True, "ok"),
])
def test_detect_gpu(monkeypatch, provider, onnx_installed, expected):
    monkeypatch.setattr(matting, "gpu_provider", lambda: provider)
    monkeypatch.setattr(matting, "have_onnx", lambda: onnx_installed)
    result = matting.detect_gpu()
    assert result["state"] == expected
    assert result["provider"] == (provider if expected in ("ok", "no_onnx") else None)
