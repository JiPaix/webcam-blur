"""Person matting (CPU, onnxruntime) and blur on the public-domain test portrait. The GPU
path (ncnn, Vulkan) is in test_ncnn.py."""


import numpy as np
import pytest
from conftest import BACKGROUND, PERSON, TEXTURED, detail, mean_abs_diff, region

from webcam_blur import matting
from webcam_blur.stream import MAX_INTERVAL, adapt_interval, blur_background, matting_interval

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


@pytest.mark.parametrize("quality", ["low", "standard", "high"])
def test_cpu_matting(frame, quality):
    model = matting.QUALITY_MODELS[quality]
    matter = matting.CpuMatter(matting.CpuMatter.create_session(model=model), matting.CPU_WIDTH[quality], model)
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
    (30, "low", 1), (29.8, "low", 1), (60, "low", 2), (59.3, "low", 2), (50, "low", 1), (15, "low", 1),
    (90, "low", 3), (60, "standard", 1), (60, "high", 1),
])
def test_matting_interval(fps, quality, expected):
    assert matting_interval(fps, quality) == expected


def test_slow_model_runs_less_often():
    # "high" on a slow CPU at 60 fps: 25 ms model + 5 ms rest per frame, busy all the time.
    n = 1
    for _ in range(3):
        n = adapt_interval(n, 1, load=0.98, model_time=0.025, rest_time=0.005, fps=60)
    assert n == 4
    assert adapt_interval(MAX_INTERVAL, 1, 0.99, 0.2, 0.005, 60) == MAX_INTERVAL


def test_interval_comes_back_when_there_is_room():
    # Switched to a lighter quality: 4 ms model now, plenty of time per frame.
    assert adapt_interval(4, 1, load=0.3, model_time=0.004, rest_time=0.004, fps=60) == 3
    assert adapt_interval(1, 1, load=0.3, model_time=0.004, rest_time=0.004, fps=60) == 1
    assert adapt_interval(2, 2, load=0.3, model_time=0.004, rest_time=0.004, fps=60) == 2  # quality's own
    # Just keeping up: going back would overload again.
    assert adapt_interval(2, 1, load=0.6, model_time=0.012, rest_time=0.006, fps=60) == 2
    # Busy but keeping up (e.g. "standard" at 60 fps on a fast CPU: 89 % of the time): no change.
    assert adapt_interval(1, 1, load=0.89, model_time=0.011, rest_time=0.004, fps=60) == 1
    # ...and back after a busy moment pushed it up.
    assert adapt_interval(2, 1, load=0.5, model_time=0.011, rest_time=0.004, fps=60) == 1


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
