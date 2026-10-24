"""Kernel ABI constants and pure helpers (no device needed)."""

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


def test_pick_mode_prefers_720p_mjpg():
    modes = {"YUYV": [(640, 480), (1280, 720)], "MJPG": [(640, 480), (1280, 720), (1920, 1080)]}
    assert v4l2.pick_mode(modes) == ("MJPG", 1280, 720)


def test_pick_mode_falls_back_to_largest_within_limit():
    modes = {"MJPG": [(320, 240), (640, 480), (1920, 1080)]}
    assert v4l2.pick_mode(modes) == ("MJPG", 640, 480)
    assert v4l2.pick_mode({"YUYV": [(640, 360), (800, 600)]}) == ("YUYV", 800, 600)


def test_pick_mode_without_known_formats():
    assert v4l2.pick_mode({}) == ("MJPG", 640, 480)
