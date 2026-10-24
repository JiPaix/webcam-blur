import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Keep test caches (frozen GPU models, MIGraphX files) away from the user's real cache.
# Must happen before webcam_blur is imported: paths are read at import time.
if "XDG_CACHE_HOME" not in os.environ:
    _cache = tempfile.mkdtemp(prefix="webcam-blur-test-cache-")
    os.environ["XDG_CACHE_HOME"] = _cache
    atexit.register(shutil.rmtree, _cache, ignore_errors=True)
# Qt widgets (the settings window) without a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Test the checkout, unless CI asks to test the installed package.
if not os.environ.get("WEBCAM_BLUR_TEST_INSTALLED"):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from webcam_blur import i18n  # noqa: E402
from webcam_blur import models as model_files  # noqa: E402

i18n.setup("en")  # also for module-scoped fixtures, which run before the autouse one below


@pytest.fixture(scope="session")
def qt_app():
    # One QApplication for the whole run: a process cannot switch from a QCoreApplication.
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])
DATA = Path(__file__).parent / "data"
W, H = 1280, 720


def compose_test_frame():
    """1280x720 webcam-like frame: the portrait centred, fine checkerboard on both sides."""
    portrait = cv2.imread(str(DATA / "portrait.jpg"))
    assert portrait is not None and portrait.shape[0] == H
    yy, xx = np.mgrid[0:H, 0:W]
    frame = np.where(((yy // 16 + xx // 16) % 2)[..., None] == 0, 60, 200).astype(np.uint8)
    frame = np.repeat(frame, 3, axis=2) if frame.shape[2] == 1 else frame
    x0 = (W - portrait.shape[1]) // 2
    frame[:, x0:x0 + portrait.shape[1]] = portrait
    return frame, x0


FRAME, X0 = compose_test_frame()

# Regions of the test frame, as (y0, y1, x0, x1).
REGIONS = {
    "face": (170, 320, X0 + 230, X0 + 370),
    "shirt": (450, 650, X0 + 150, X0 + 330),
    "flag": (20, 300, X0 + 0, X0 + 90),
    "wall": (30, 150, X0 + 450, X0 + 560),
    "left_side": (0, H, 0, X0 - 20),
    "right_side": (0, H, X0 + 596, W),
}
PERSON = ("face", "shirt")
BACKGROUND = ("flag", "wall", "left_side", "right_side")
TEXTURED = ("flag", "left_side", "right_side")  # background regions with detail to blur


def region(img, name):
    y0, y1, x0, x1 = REGIONS[name]
    return img[y0:y1, x0:x1]


def detail(img, name):
    """Mean gradient magnitude: drops steadily as a region gets more blurred."""
    grey = cv2.cvtColor(region(img, name), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.abs(cv2.Sobel(grey, cv2.CV_32F, 1, 0)).mean() + np.abs(cv2.Sobel(grey, cv2.CV_32F, 0, 1)).mean())


def mean_abs_diff(a, b, name):
    return float(np.abs(region(a, name).astype(np.int16) - region(b, name).astype(np.int16)).mean())


@pytest.fixture(scope="session")
def frame():
    return FRAME.copy()


@pytest.fixture(scope="session")
def models():
    model_files.ensure_models()


@pytest.fixture(autouse=True)
def english():
    i18n.setup("en")
