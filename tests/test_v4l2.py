"""Kernel ABI constants and pure helpers (no device needed)."""

import errno

import pytest

from webcam_blur import v4l2


def _ioc(direction, type_, nr, size):
    return (direction << 30) | (size << 16) | (ord(type_) << 8) | nr


IOW, IOR, IOWR = 1, 2, 3


def test_loopback_abi():
    assert v4l2._LB_CONFIG.size == 72
    assert v4l2.LOOPBACK_CTL_ADD[0] == _ioc(IOW, "~", 1, 72)
    assert v4l2.LOOPBACK_CTL_REMOVE[0] == _ioc(IOW, "~", 2, 4)
    assert v4l2.LOOPBACK_CTL_QUERY[0] == _ioc(IOWR, "~", 3, 72)
    # v4l2loopback 0.13 used plain constants
    assert [r[1] for r in (v4l2.LOOPBACK_CTL_ADD, v4l2.LOOPBACK_CTL_REMOVE, v4l2.LOOPBACK_CTL_QUERY)] == \
        [0x4C80, 0x4C81, 0x4C82]


def test_v4l2_abi():
    assert v4l2.VIDIOC_S_FMT == _ioc(IOWR, "V", 5, 208)
    assert v4l2.VIDIOC_ENUM_FMT == _ioc(IOWR, "V", 2, 64)
    assert v4l2.VIDIOC_ENUM_FRAMESIZES == _ioc(IOWR, "V", 74, 44)
    assert v4l2.VIDIOC_ENUM_FRAMEINTERVALS == _ioc(IOWR, "V", 75, 52)
    assert v4l2.VIDIOC_S_PARM == _ioc(IOWR, "V", 22, 204)


def modes_of(*specs):
    return v4l2.sort_modes(v4l2.Mode(w, h, fps, fmt) for w, h, fps, fmt in specs)


C922_LIKE = modes_of((1920, 1080, 30, "MJPG"), (1920, 1080, 15, "MJPG"), (1280, 720, 60, "MJPG"),
                     (1280, 720, 30, "MJPG"), (1280, 720, 10, "YUYV"), (640, 480, 30, "MJPG"),
                     (2304, 1536, 2, "YUYV"), (160, 90, 30, "MJPG"))


def test_auto_mode_prefers_720p_at_30fps():
    assert v4l2.auto_mode(C922_LIKE) == v4l2.Mode(1280, 720, 30, "MJPG")


def test_auto_mode_for_a_4k20_1080p60_camera():
    cam = modes_of((3840, 2160, 20, "MJPG"), (1920, 1080, 60, "MJPG"), (1920, 1080, 30, "MJPG"),
                   (640, 480, 30, "YUYV"))
    assert v4l2.auto_mode(cam) == v4l2.Mode(1920, 1080, 30, "MJPG")  # no 720p: next size up, at 30 fps


def test_auto_mode_falls_back_to_fastest_rate():
    slow = modes_of((800, 600, 15, "YUYV"), (640, 480, 20, "YUYV"))
    assert v4l2.auto_mode(slow) == v4l2.Mode(800, 600, 15, "YUYV")  # nothing reaches 30 fps: largest size
    assert v4l2.auto_mode([]) == v4l2.Mode(640, 480, 30.0)


def test_menu_modes_keeps_useful_combinations():
    keys = [m.key() for m in v4l2.menu_modes(C922_LIKE)]
    assert keys == [(1920, 1080, 30), (1280, 720, 60), (1280, 720, 30), (640, 480, 30)]
    cam = modes_of((3840, 2160, 20, "MJPG"), (1920, 1080, 60, "MJPG"), (1920, 1080, 30, "MJPG"),
                   (1920, 1080, 24, "MJPG"))
    assert [m.key() for m in v4l2.menu_modes(cam)] == [(3840, 2160, 20), (1920, 1080, 60), (1920, 1080, 30)]


def test_find_mode_matches_saved_choice():
    assert v4l2.find_mode(C922_LIKE, {"width": 1280, "height": 720, "fps": 60}) == v4l2.Mode(1280, 720, 60, "MJPG")
    assert v4l2.find_mode(C922_LIKE, {"width": 1280, "height": 720, "fps": 50}) is None  # not offered: automatic
    assert v4l2.find_mode(C922_LIKE, None) is None


class FakeIoctl:
    """Simulates a v4l2loopback control device that only knows one set of request numbers."""

    def __init__(self, legacy_driver):
        self.known = {r[1] if legacy_driver else r[0]
                      for r in (v4l2.LOOPBACK_CTL_ADD, v4l2.LOOPBACK_CTL_REMOVE, v4l2.LOOPBACK_CTL_QUERY)}
        self.calls = []

    def __call__(self, fd, request, arg=0, mutate=True):
        self.calls.append(request)
        if request not in self.known:
            raise OSError(errno.ENOTTY, "Inappropriate ioctl for device")
        return 7


def control_with(monkeypatch, fake):
    monkeypatch.setattr(v4l2.fcntl, "ioctl", fake)
    monkeypatch.setattr(v4l2, "_wait_node", lambda path: None)
    ctl = object.__new__(v4l2.LoopbackControl)
    ctl.fd, ctl.legacy = -1, None
    return ctl


@pytest.mark.parametrize("legacy_driver", [False, True], ids=["current", "0.13-0.14"])
def test_control_requests_for_each_driver_generation(monkeypatch, legacy_driver):
    fake = FakeIoctl(legacy_driver)
    ctl = control_with(monkeypatch, fake)
    assert ctl.add("Cam (blur)") == 7
    assert ctl.legacy is legacy_driver
    assert ctl.remove(7) is True
    # Once detected, the right generation is used directly, without a failing first try.
    assert fake.calls[-1] == v4l2.LOOPBACK_CTL_REMOVE[1 if legacy_driver else 0]
    assert len(fake.calls) == (3 if legacy_driver else 2)


def test_control_does_not_mask_real_errors(monkeypatch):
    def busy(fd, request, arg=0, mutate=True):
        raise OSError(errno.EBUSY, "Device or resource busy")
    ctl = control_with(monkeypatch, busy)
    ctl.legacy = False
    assert ctl.remove(7) is False  # EBUSY: still open somewhere
    with pytest.raises(OSError):
        ctl.add("Cam (blur)")


# --- Webcam controls ---------------------------------------------------------------

def test_control_ioctl_layouts():
    assert v4l2._QUERYCTRL.size == 68 and v4l2.VIDIOC_QUERYCTRL == 0xC0445624
    assert v4l2._QUERYMENU.size == 44 and v4l2.VIDIOC_QUERYMENU == 0xC02C5625
    assert v4l2._CONTROL.size == 8


@pytest.mark.parametrize("name, key", [
    ("Exposure Time, Absolute", "exposure_time_absolute"),
    ("Exposure, Dynamic Framerate", "exposure_dynamic_framerate"),
    ("White Balance, Automatic", "white_balance_automatic"),
    ("Brightness", "brightness"),
])
def test_control_key(name, key):
    assert v4l2.control_key(name) == key


def test_apply_controls_switches_auto_modes_first(monkeypatch):
    def ctl(cid, key, ctype):
        return v4l2.Control(cid, key, key, ctype, 0, 1000, 1, 0, 0, 0, {})
    monkeypatch.setattr(v4l2, "list_controls", lambda path: [
        ctl(1, "exposure_time_absolute", v4l2.CTRL_INTEGER), ctl(2, "auto_exposure", v4l2.CTRL_MENU),
        ctl(3, "focus_absolute", v4l2.CTRL_INTEGER), ctl(4, "focus_automatic_continuous", v4l2.CTRL_BOOLEAN)])
    done = []
    monkeypatch.setattr(v4l2, "set_control", lambda path, cid, value: done.append(cid))
    failed = v4l2.apply_controls("/dev/null", {"exposure_time_absolute": 400, "focus_absolute": 30,
                                               "auto_exposure": 1, "focus_automatic_continuous": 0,
                                               "gone": 1})
    assert set(done[:2]) == {2, 4} and set(done[2:]) == {1, 3}, "auto switches before manual values"
    assert failed == ["gone"]
