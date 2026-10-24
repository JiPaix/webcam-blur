"""Kernel interfaces: v4l2loopback control device, loopback output, webcam formats."""

import errno
import fcntl
import os
import struct
import time

import cv2
import numpy as np


def fourcc(s):
    return struct.unpack("<I", s.encode())[0]


# --- /dev/v4l2loopback ---------------------------------------------------------
# struct v4l2_loopback_config (v4l2loopback.h, 0.12+): 72 bytes.
_LB_CONFIG = struct.Struct("<ii32sIIIIiiii")
LOOPBACK_CTL_ADD = 0x40487E01     # _IOW('~', 1, struct v4l2_loopback_config)
LOOPBACK_CTL_REMOVE = 0x40047E02  # _IOW('~', 2, __u32)
LOOPBACK_CTL_QUERY = 0xC0487E03   # _IOWR('~', 3, struct v4l2_loopback_config)
LABEL_MAX = 31  # card_label is char[32], NUL-terminated


def fit_label(template, name):
    """Format ``template`` with ``name`` shortened so the label fits v4l2loopback's 31 bytes."""
    label = template.format(name=name)
    while len(label.encode()) > LABEL_MAX and name:
        name = name[:-1].rstrip()
        label = template.format(name=name + "…")
    return label


class LoopbackControl:
    """Creates/removes v4l2loopback devices at runtime (needs access to /dev/v4l2loopback)."""

    def __init__(self, path="/dev/v4l2loopback"):
        self.fd = os.open(path, os.O_RDWR)

    def add(self, label):
        """Create a capture-exclusive loopback device, return its /dev/videoN number."""
        buf = bytearray(_LB_CONFIG.pack(-1, -1, label.encode()[:LABEL_MAX], 0, 0, 0, 0, 0, 0, 0, 0))
        nr = fcntl.ioctl(self.fd, LOOPBACK_CTL_ADD, buf, True)
        _wait_node(f"/dev/video{nr}")
        return nr

    def remove(self, nr):
        """Remove a loopback device. Returns False if it is still open somewhere (EBUSY)."""
        try:
            fcntl.ioctl(self.fd, LOOPBACK_CTL_REMOVE, nr)
            return True
        except OSError as e:
            if e.errno == errno.EBUSY:
                return False
            if e.errno in (errno.ENODEV, errno.EINVAL, errno.ENOENT):
                return True  # already gone
            raise

    def query(self, nr):
        """Label of loopback device ``nr``, or None if it is not a loopback device."""
        buf = bytearray(_LB_CONFIG.pack(nr, -1, b"", 0, 0, 0, 0, 0, 0, 0, 0))
        try:
            fcntl.ioctl(self.fd, LOOPBACK_CTL_QUERY, buf, True)
        except OSError:
            return None
        return _LB_CONFIG.unpack(buf)[2].split(b"\0")[0].decode(errors="replace")

    def devices(self):
        """{nr: label} of all existing loopback devices."""
        found = {}
        for entry in os.listdir("/sys/class/video4linux"):
            if entry.startswith("video") and entry[5:].isdigit():
                label = self.query(int(entry[5:]))
                if label is not None:
                    found[int(entry[5:])] = label
        return found

    def close(self):
        os.close(self.fd)


def _wait_node(path, timeout=5.0):
    """udev creates the node and applies permissions asynchronously after CTL_ADD."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.access(path, os.R_OK | os.W_OK):
            return
        time.sleep(0.05)
    raise TimeoutError(f"{path} did not appear")


# --- Loopback output ----------------------------------------------------------

VIDIOC_S_FMT = 0xC0D05605  # _IOWR('V', 5, struct v4l2_format), sizeof == 208
V4L2_BUF_TYPE_VIDEO_OUTPUT = 2
V4L2_FIELD_NONE = 1
V4L2_COLORSPACE_SRGB = 8

# Can OpenCV produce YUYV? Otherwise fall back to I420.
_YUYV_CODE = getattr(cv2, "COLOR_BGR2YUV_YUY2", None)


class LoopbackOutput:
    def __init__(self, path, width, height):
        self.width, self.height = width, height
        self.fd = os.open(path, os.O_RDWR)
        if _YUYV_CODE is not None:
            pixfmt, bpl, size = fourcc("YUYV"), width * 2, width * height * 2
        else:
            pixfmt, bpl, size = fourcc("YU12"), width, width * height * 3 // 2
        fmt = struct.pack("<I4x12I", V4L2_BUF_TYPE_VIDEO_OUTPUT, width, height, pixfmt,
                          V4L2_FIELD_NONE, bpl, size, V4L2_COLORSPACE_SRGB, 0, 0, 0, 0, 0)
        fmt += b"\0" * (208 - len(fmt))
        fcntl.ioctl(self.fd, VIDIOC_S_FMT, fmt)
        self.black = self.convert(np.zeros((height, width, 3), np.uint8))

    def convert(self, bgr):
        if _YUYV_CODE is not None:
            return cv2.cvtColor(bgr, _YUYV_CODE)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)

    def write(self, bgr):
        os.write(self.fd, self.convert(bgr).tobytes())

    def write_black(self):
        os.write(self.fd, self.black.tobytes())

    def close(self):
        os.close(self.fd)


def readers_of(path, own_pid=os.getpid()):
    """Number of processes (other than us) that have ``path`` open."""
    target = os.path.realpath(path)
    count = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == own_pid:
            continue
        fd_dir = f"/proc/{pid}/fd"
        try:
            for fd in os.listdir(fd_dir):
                try:
                    if os.readlink(f"{fd_dir}/{fd}") == target:
                        count += 1
                        break
                except OSError:
                    pass
        except OSError:
            pass
    return count


# --- Webcam formats (queried without streaming, so the LED stays off) ---------

VIDIOC_ENUM_FMT = 0xC0405602         # _IOWR('V', 2, struct v4l2_fmtdesc), 64 bytes
VIDIOC_ENUM_FRAMESIZES = 0xC02C564A  # _IOWR('V', 74, struct v4l2_frmsizeenum), 44 bytes
V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_FRMSIZE_TYPE_DISCRETE = 1


def capture_modes(path):
    """{fourcc: [(w, h), …]} of discrete frame sizes, for MJPG and YUYV."""
    modes = {}
    fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    try:
        for i in range(64):
            buf = bytearray(struct.pack("<III32sII12x", i, V4L2_BUF_TYPE_VIDEO_CAPTURE, 0, b"", 0, 0))
            try:
                fcntl.ioctl(fd, VIDIOC_ENUM_FMT, buf, True)
            except OSError:
                break
            pixfmt = struct.unpack_from("<I", buf, 44)[0]
            name = struct.pack("<I", pixfmt).decode(errors="replace")
            if name not in ("MJPG", "YUYV"):
                continue
            sizes = []
            for j in range(256):
                fbuf = bytearray(struct.pack("<III32x", j, pixfmt, 0))
                try:
                    fcntl.ioctl(fd, VIDIOC_ENUM_FRAMESIZES, fbuf, True)
                except OSError:
                    break
                ftype, w, h = struct.unpack_from("<III", fbuf, 8)
                if ftype != V4L2_FRMSIZE_TYPE_DISCRETE:
                    break
                sizes.append((w, h))
            modes[name] = sizes
    finally:
        os.close(fd)
    return modes


def pick_mode(modes, preferred=(1280, 720), max_width=1280):
    """Choose (fourcc, w, h): 1280x720 if offered, else the largest size up to 1280 wide.

    MJPG is preferred: at 720p, raw YUYV usually cannot reach 30 fps over USB 2.
    """
    for fmt in ("MJPG", "YUYV"):
        sizes = modes.get(fmt) or []
        if preferred in sizes:
            return fmt, *preferred
    for fmt in ("MJPG", "YUYV"):
        sizes = [s for s in modes.get(fmt) or [] if s[0] <= max_width]
        if sizes:
            w, h = max(sizes, key=lambda s: s[0] * s[1])
            return fmt, w, h
    return "MJPG", 640, 480
