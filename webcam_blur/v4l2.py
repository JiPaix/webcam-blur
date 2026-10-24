"""Kernel interfaces: v4l2loopback control device, loopback output, webcam formats."""

import errno
import fcntl
import os
import struct
import time
from dataclasses import dataclass


def fourcc(s):
    return struct.unpack("<I", s.encode())[0]


# --- /dev/v4l2loopback ---------------------------------------------------------
# The control device exists since v4l2loopback 0.13. struct v4l2_loopback_config: 72 bytes.
_LB_CONFIG = struct.Struct("<ii32sIIIIiiii")
# (current, legacy) request numbers: 0.13 used plain constants, later versions _IOW/_IOWR.
LOOPBACK_CTL_ADD = (0x40487E01, 0x4C80)     # _IOW('~', 1, struct v4l2_loopback_config)
LOOPBACK_CTL_REMOVE = (0x40047E02, 0x4C81)  # _IOW('~', 2, __u32)
LOOPBACK_CTL_QUERY = (0xC0487E03, 0x4C82)   # _IOWR('~', 3, struct v4l2_loopback_config)
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
        self.legacy = None  # unknown until the first request

    def _ioctl(self, request, arg):
        current, legacy = request
        if not self.legacy:
            try:
                result = fcntl.ioctl(self.fd, current, arg, True)
                self.legacy = False
                return result
            except OSError as e:
                if e.errno != errno.ENOTTY or self.legacy is False:
                    raise
                self.legacy = True
        return fcntl.ioctl(self.fd, legacy, arg, True)

    def add(self, label):
        """Create a capture-exclusive loopback device, return its /dev/videoN number."""
        buf = bytearray(_LB_CONFIG.pack(-1, -1, label.encode()[:LABEL_MAX], 0, 0, 0, 0, 0, 0, 0, 0))
        nr = self._ioctl(LOOPBACK_CTL_ADD, buf)
        _wait_node(f"/dev/video{nr}")
        return nr

    def remove(self, nr):
        """Remove a loopback device. Returns False if it is still open somewhere (EBUSY)."""
        try:
            self._ioctl(LOOPBACK_CTL_REMOVE, nr)
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
            self._ioctl(LOOPBACK_CTL_QUERY, buf)
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
# The tray process opens the device and sets its format; while an app reads it, the stream
# subprocess writes frames through the same file descriptor. Frames are YUYV.

VIDIOC_S_FMT = 0xC0D05605   # _IOWR('V', 5, struct v4l2_format), sizeof == 208
VIDIOC_S_PARM = 0xC0CC5616  # _IOWR('V', 22, struct v4l2_streamparm), sizeof == 204
V4L2_BUF_TYPE_VIDEO_OUTPUT = 2
V4L2_FIELD_NONE = 1
V4L2_COLORSPACE_SRGB = 8
YUYV_BLACK = bytes((16, 128))  # one pixel (BT.601, as OpenCV's BGR2YUV_YUY2)


class LoopbackOutput:
    def __init__(self, path, width, height):
        self._setup(os.open(path, os.O_RDWR), width, height)
        fmt = struct.pack("<I4x12I", V4L2_BUF_TYPE_VIDEO_OUTPUT, width, height, fourcc("YUYV"),
                          V4L2_FIELD_NONE, width * 2, width * height * 2, V4L2_COLORSPACE_SRGB, 0, 0, 0, 0, 0)
        fmt += b"\0" * (208 - len(fmt))
        fcntl.ioctl(self.fd, VIDIOC_S_FMT, fmt)

    @classmethod
    def attach(cls, fd, width, height):
        """Write to a device already opened and configured by another process."""
        out = cls.__new__(cls)
        out._setup(fd, width, height)
        return out

    def _setup(self, fd, width, height):
        self.fd, self.width, self.height = fd, width, height
        self.black = YUYV_BLACK * (width * height)

    def set_fps(self, fps):
        """Advertise the frame rate to apps (best effort: older drivers may refuse)."""
        den = round(fps * 1000)
        parm = struct.pack("<IIIIIII4I", V4L2_BUF_TYPE_VIDEO_OUTPUT, 0, 0, 1000, den, 0, 0, 0, 0, 0, 0)
        parm += b"\0" * (204 - len(parm))
        try:
            fcntl.ioctl(self.fd, VIDIOC_S_PARM, parm)
        except OSError:
            pass

    def write(self, bgr):
        """Write one BGR frame (stream subprocess only: needs OpenCV)."""
        os.write(self.fd, bgr_to_yuyv(bgr))

    def write_black(self):
        os.write(self.fd, self.black)

    def close(self):
        os.close(self.fd)


def bgr_to_yuyv(bgr):
    import cv2  # imported here: the tray process never loads OpenCV
    code = getattr(cv2, "COLOR_BGR2YUV_YUY2", None)
    if code is not None:
        return cv2.cvtColor(bgr, code).tobytes()
    import numpy as np  # OpenCV < 4.7: pack YUYV by hand
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    yuyv = np.empty((bgr.shape[0], bgr.shape[1], 2), np.uint8)
    yuyv[..., 0] = yuv[..., 0]
    yuyv[:, 0::2, 1] = ((yuv[:, 0::2, 1].astype(np.uint16) + yuv[:, 1::2, 1]) // 2)
    yuyv[:, 1::2, 1] = ((yuv[:, 0::2, 2].astype(np.uint16) + yuv[:, 1::2, 2]) // 2)
    return yuyv.tobytes()


def readers_of(path, exclude=()):
    """Number of processes, other than us and ``exclude``, that have ``path`` open."""
    skip = {os.getpid(), *exclude}
    target = os.path.realpath(path)
    count = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) in skip:
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


# --- Webcam modes (queried without streaming, so the LED stays off) -----------

VIDIOC_ENUM_FMT = 0xC0405602             # _IOWR('V', 2, struct v4l2_fmtdesc), 64 bytes
VIDIOC_ENUM_FRAMESIZES = 0xC02C564A      # _IOWR('V', 74, struct v4l2_frmsizeenum), 44 bytes
VIDIOC_ENUM_FRAMEINTERVALS = 0xC034564B  # _IOWR('V', 75, struct v4l2_frmivalenum), 52 bytes
V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_FRMSIZE_TYPE_DISCRETE = 1
V4L2_FRMIVAL_TYPE_DISCRETE = 1
FORMATS = ("MJPG", "YUYV")  # in order of preference: raw YUYV rarely reaches 30 fps at 720p over USB 2


@dataclass(frozen=True)
class Mode:
    width: int
    height: int
    fps: float
    fourcc: str = "MJPG"

    def key(self):
        """Identity of the mode as the user sees it (the pixel format is an implementation detail)."""
        return self.width, self.height, round(self.fps, 2)

    def to_config(self):
        return {"width": self.width, "height": self.height, "fps": round(self.fps, 2)}


def _ioctl_iter(fd, request, make, limit=256):
    for i in range(limit):
        buf = bytearray(make(i))
        try:
            fcntl.ioctl(fd, request, buf, True)
        except OSError:
            return
        yield buf


def _sizes(fd, pixfmt, wanted=(1280, 720)):
    sizes = []
    for buf in _ioctl_iter(fd, VIDIOC_ENUM_FRAMESIZES, lambda i: struct.pack("<III32x", i, pixfmt, 0)):
        ftype, w, h = struct.unpack_from("<III", buf, 8)
        if ftype == V4L2_FRMSIZE_TYPE_DISCRETE:
            sizes.append((w, h))
            continue
        # Size range (stepwise/continuous): offer the usual size if it fits, and the largest.
        min_w, max_w, step_w, min_h, max_h, step_h = struct.unpack_from("<6I", buf, 12)
        ww, wh = wanted
        if min_w <= ww <= max_w and min_h <= wh <= max_h \
                and (ww - min_w) % max(step_w, 1) == 0 and (wh - min_h) % max(step_h, 1) == 0:
            sizes.append(wanted)
        sizes.append((max_w, max_h))
        break
    return sizes


def _rates(fd, pixfmt, w, h):
    rates = []
    for buf in _ioctl_iter(fd, VIDIOC_ENUM_FRAMEINTERVALS, lambda i: struct.pack("<IIII36x", i, pixfmt, w, h)):
        ftype, num, den = struct.unpack_from("<III", buf, 16)
        if ftype == V4L2_FRMIVAL_TYPE_DISCRETE:
            if num and den:
                rates.append(den / num)
            continue
        # Interval range: offer the fastest rate, and 30 fps if it is within the range.
        min_n, min_d, max_n, max_d = struct.unpack_from("<4I", buf, 20)
        fastest, slowest = (min_d / min_n if min_n else 30.0), (max_d / max_n if max_n else 1.0)
        rates.append(fastest)
        if slowest <= 30.0 < fastest:
            rates.append(30.0)
        break
    return rates or [30.0]  # some drivers do not report intervals


def list_modes(path):
    """Every resolution x frame rate the webcam offers in MJPG or YUYV, best first.

    When both formats offer the same resolution and rate, MJPG is kept.
    """
    found = {}
    fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    try:
        formats = {}
        for buf in _ioctl_iter(fd, VIDIOC_ENUM_FMT,
                               lambda i: struct.pack("<III32sII12x", i, V4L2_BUF_TYPE_VIDEO_CAPTURE, 0, b"", 0, 0),
                               limit=64):
            pixfmt = struct.unpack_from("<I", buf, 44)[0]
            formats[struct.pack("<I", pixfmt).decode(errors="replace")] = pixfmt
        for name in FORMATS:
            if name not in formats:
                continue
            for w, h in _sizes(fd, formats[name]):
                for fps in _rates(fd, formats[name], w, h):
                    mode = Mode(w, h, fps, name)
                    found.setdefault(mode.key(), mode)
    finally:
        os.close(fd)
    return sort_modes(found.values())


def sort_modes(modes):
    return sorted(modes, key=lambda m: (m.width * m.height, m.fps), reverse=True)


def auto_mode(modes, preferred=(1280, 720)):
    """Default mode: 1280x720 at 30 fps if offered; otherwise the smallest size at least as
    wide that reaches 30 fps (e.g. 1080p30), or else the largest one below it.

    Video-call apps send 720p at most in practice, and every extra pixel costs processing.
    """
    if not modes:
        return Mode(640, 480, 30.0)

    def best_rate(candidates):
        fast = [m for m in candidates if m.fps >= 29.5]
        if fast:  # 30 fps rather than 60: calling apps send 30, and it halves the work
            return min(fast, key=lambda m: (m.fps, FORMATS.index(m.fourcc)))
        return max(candidates, key=lambda m: m.fps)

    pool = [m for m in modes if m.fps >= 29.5] or modes
    sizes = {(m.width, m.height) for m in pool}
    if preferred in sizes:
        size = preferred
    else:
        larger = [sz for sz in sizes if sz[0] >= preferred[0]]
        size = min(larger, key=lambda sz: sz[0] * sz[1]) if larger else max(sizes, key=lambda sz: sz[0] * sz[1])
    return best_rate([m for m in modes if (m.width, m.height) == size])


def find_mode(modes, wanted):
    """The listed mode matching a saved ``{"width", "height", "fps"}`` choice, or None."""
    if not wanted:
        return None
    key = (wanted.get("width"), wanted.get("height"), round(float(wanted.get("fps", 0)), 2))
    return next((m for m in modes if m.key() == key), None)


def menu_modes(modes, min_fps=15.0, min_width=320):
    """The modes worth offering in a menu: for each size, its fastest rate plus 60 and
    30 fps when available. Sizes that cannot reach ``min_fps`` or narrower than
    ``min_width`` are left out (unless that would leave nothing)."""
    by_size = {}
    for m in modes:
        by_size.setdefault((m.width, m.height), []).append(m)
    keep = []
    for (w, _h), size_modes in by_size.items():
        fastest = max(size_modes, key=lambda m: m.fps)
        if fastest.fps < min_fps or w < min_width:
            continue
        for m in size_modes:
            if m is fastest or round(m.fps) in (30, 60):
                keep.append(m)
    return sort_modes(keep or modes)


# --- Webcam controls (brightness, exposure, focus…) --------------------------------

VIDIOC_QUERYCTRL = 0xC0445624  # _IOWR('V', 36, struct v4l2_queryctrl), 68 bytes
VIDIOC_QUERYMENU = 0xC02C5625  # _IOWR('V', 37, struct v4l2_querymenu), 44 bytes (packed)
VIDIOC_G_CTRL = 0xC008561B     # _IOWR('V', 27, struct v4l2_control), 8 bytes
VIDIOC_S_CTRL = 0xC008561C     # _IOWR('V', 28, struct v4l2_control)
_QUERYCTRL = struct.Struct("<II32siiiiI8x")
_QUERYMENU = struct.Struct("<II32s4x")
_CONTROL = struct.Struct("<Ii")
CTRL_FLAG_NEXT = 0x80000000
CTRL_FLAG_DISABLED, CTRL_FLAG_READ_ONLY, CTRL_FLAG_INACTIVE = 0x01, 0x04, 0x10
CTRL_INTEGER, CTRL_BOOLEAN, CTRL_MENU = 1, 2, 3
CONTROL_TYPES = (CTRL_INTEGER, CTRL_BOOLEAN, CTRL_MENU)  # the ones offered in the settings window


def control_key(name):
    """Stable name of a control, as v4l2-ctl prints it ("Exposure Time, Absolute" ->
    "exposure_time_absolute"): used in the config and for translations."""
    key = "".join(c if c.isalnum() else "_" for c in name.lower())
    while "__" in key:
        key = key.replace("__", "_")
    return key.strip("_")


@dataclass
class Control:
    id: int
    key: str
    name: str        # as the driver reports it (English)
    type: int
    minimum: int
    maximum: int
    step: int
    default: int
    value: int
    flags: int
    menu: dict       # menu controls: {index: driver name}

    @property
    def inactive(self):
        """Ignored by the webcam in its current state (e.g. manual exposure while auto)."""
        return bool(self.flags & (CTRL_FLAG_INACTIVE | CTRL_FLAG_READ_ONLY))


def _menu(fd, ctrl_id, minimum, maximum):
    items = {}
    for index in range(minimum, maximum + 1):
        buf = bytearray(_QUERYMENU.pack(ctrl_id, index, b""))
        try:
            fcntl.ioctl(fd, VIDIOC_QUERYMENU, buf, True)
        except OSError:
            continue  # holes are allowed
        items[index] = _QUERYMENU.unpack(buf)[2].split(b"\0", 1)[0].decode(errors="replace")
    return items


def list_controls(path):
    """The webcam's integer, on/off and menu controls with their current values."""
    controls = []
    fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    try:
        next_id = CTRL_FLAG_NEXT
        while True:
            buf = bytearray(_QUERYCTRL.pack(next_id, 0, b"", 0, 0, 0, 0, 0))
            try:
                fcntl.ioctl(fd, VIDIOC_QUERYCTRL, buf, True)
            except OSError:
                break  # EINVAL: no more controls
            cid, ctype, name, lo, hi, step, default, flags = _QUERYCTRL.unpack(buf)
            next_id = cid | CTRL_FLAG_NEXT
            if ctype not in CONTROL_TYPES or flags & CTRL_FLAG_DISABLED:
                continue
            name = name.split(b"\0", 1)[0].decode(errors="replace")
            try:
                value = _get(fd, cid)
            except OSError:
                continue
            menu = _menu(fd, cid, lo, hi) if ctype == CTRL_MENU else {}
            controls.append(Control(cid, control_key(name), name, ctype, lo, hi, max(step, 1),
                                    default, value, flags, menu))
    finally:
        os.close(fd)
    return controls


def _get(fd, ctrl_id):
    buf = bytearray(_CONTROL.pack(ctrl_id, 0))
    fcntl.ioctl(fd, VIDIOC_G_CTRL, buf, True)
    return _CONTROL.unpack(buf)[1]


def set_control(path, ctrl_id, value):
    fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    try:
        fcntl.ioctl(fd, VIDIOC_S_CTRL, bytearray(_CONTROL.pack(ctrl_id, value)), True)
    finally:
        os.close(fd)


def apply_controls(path, saved):
    """Set the saved values ({key: value}) on the webcam. On/off and menu controls go first:
    they switch automatic modes. Manual values that their automatic mode then makes inactive
    (e.g. an exposure time saved before exposure went back to automatic) are skipped.
    Returns the keys that could not be set."""
    controls = {c.key: c for c in list_controls(path)}
    failed = [k for k in saved if k not in controls]
    for integers in (False, True):
        if integers:
            controls = {c.key: c for c in list_controls(path)}  # flags after the mode switches
        for key in saved:
            c = controls.get(key)
            if c is None or (c.type == CTRL_INTEGER) != integers or (integers and c.inactive):
                continue
            try:
                set_control(path, c.id, int(saved[key]))
            except OSError:
                failed.append(key)
    return failed
