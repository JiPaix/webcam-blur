"""Person matting on the GPU: RVM on ncnn with Vulkan, through ncnn's C API (ctypes).

Vulkan runs on AMD, NVIDIA and Intel GPUs with the system's own driver, and needs far
less memory than ROCm or CUDA (about 60-190 MB of VRAM, and a few tens of MB of RAM).

The models are converted from RVM's ONNX files by tools/convert_ncnn.py, at 640x360.
Their weights work at any size; only a few numbers in the network description depend
on it (the downsampling ratio, crops, the output size), so a description is generated
for each camera resolution from that template (see sized_param).
"""

import ctypes
import math
import re
import threading

import cv2
import numpy as np

from .models import CACHE, model_path

LIBRARY = "libncnn.so.1"
TEMPLATE_INTERNAL = (180, 320)  # the recurrent part's input size in the 640x360 templates
TARGET = 320  # RVM runs its recurrent part at about this size on the long side
PIXEL_BGR2RGB = 2 | (1 << 16)
NORM = (ctypes.c_float * 3)(1 / 255, 1 / 255, 1 / 255)

_lib = None


def library():
    """ncnn's C API, or None if ncnn is not installed."""
    global _lib
    if _lib is None:
        try:
            lib = ctypes.CDLL(LIBRARY)
        except OSError:
            return None
        P, Int, F = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
        for name, res, args in [
            ("ncnn_option_create", P, []),
            ("ncnn_option_destroy", None, [P]),
            ("ncnn_option_set_use_vulkan_compute", None, [P, Int]),
            ("ncnn_option_set_num_threads", None, [P, Int]),
            ("ncnn_net_create", P, []),
            ("ncnn_net_destroy", None, [P]),
            ("ncnn_net_set_option", None, [P, P]),
            ("ncnn_net_load_param", Int, [P, ctypes.c_char_p]),
            ("ncnn_net_load_model", Int, [P, ctypes.c_char_p]),
            ("ncnn_extractor_create", P, [P]),
            ("ncnn_extractor_destroy", None, [P]),
            ("ncnn_extractor_input", Int, [P, ctypes.c_char_p, P]),
            ("ncnn_extractor_extract", Int, [P, ctypes.c_char_p, ctypes.POINTER(P)]),
            ("ncnn_mat_create_3d", P, [Int, Int, Int, P]),
            ("ncnn_mat_destroy", None, [P]),
            ("ncnn_mat_fill_float", None, [P, F]),
            ("ncnn_mat_get_w", Int, [P]),
            ("ncnn_mat_get_h", Int, [P]),
            ("ncnn_mat_get_data", P, [P]),
            ("ncnn_mat_from_pixels_resize", P, [ctypes.c_char_p, Int, Int, Int, Int, Int, Int, P]),
            ("ncnn_mat_substract_mean_normalize", None, [P, P, ctypes.POINTER(F)]),
        ]:
            f = getattr(lib, name)
            f.restype, f.argtypes = res, args
        # Not in the C API: ncnn::get_gpu_count(), a plain "int f()" C++ function.
        lib.gpu_count = getattr(lib, "_ZN4ncnn13get_gpu_countEv", None)
        if lib.gpu_count is not None:
            lib.gpu_count.restype, lib.gpu_count.argtypes = Int, []
        _lib = lib
    return _lib


def gpu_count():
    lib = library()
    return lib.gpu_count() if lib is not None and lib.gpu_count is not None else 0


def model_files(model):
    """(description template, weights) of an RVM ONNX model's ncnn conversion."""
    stem = model.removesuffix("_fp32.onnx")
    return model_path(f"{stem}.ncnn.param"), model_path(f"{stem}.ncnn.bin")


def _levels(n):
    """Sizes after each stride-2 stage: n/2, n/4, n/8, n/16, rounded up."""
    out = []
    for _ in range(4):
        n = math.ceil(n / 2)
        out.append(n)
    return out


def sized_param(template, width, height):
    """The network description for ``width``x``height`` frames, and the recurrent state
    shapes (channels, height, width), from a template converted at 640x360."""
    ratio = min(1.0, TARGET / max(width, height))
    hs, ws = _levels(math.floor(height * ratio)), _levels(math.floor(width * ratio))
    t_hs, t_ws = _levels(TEMPLATE_INTERNAL[0]), _levels(TEMPLATE_INTERNAL[1])
    lines, channels = [], {}
    for line in template.splitlines():
        kind = line.split(maxsplit=1)[0] if line.strip() else ""
        if kind == "Input":  # template shapes, written by tools/convert_ncnn.py
            name = line.split()[1]
            c = re.search(r" 2=(\d+)", line)
            if name != "in0" and c:
                channels[name] = int(c.group(1))
            line = re.sub(r" 0=\d+ 1=\d+ 2=\d+", "", line)  # sizes come from the inputs at run time
        scale = re.search(r" 1=([0-9.e+-]+) 2=", line)
        if kind == "Interp" and scale and float(scale.group(1)) < 1:  # downsampling of the input
            line = re.sub(r" 1=[0-9.e+-]+ 2=[0-9.e+-]+", f" 1={ratio:.8g} 2={ratio:.8g}", line)
        elif kind == "Interp" and " 3=" in line:  # back to the frame's resolution
            line = re.sub(r" 3=\d+ 4=\d+", f" 3={height} 4={width}", line)
        elif kind == "Crop":  # decoder features cropped to the encoder's size at that level
            m = re.search(r"-23310=2,(\d+),(\d+)", line)
            level = list(zip(t_hs, t_ws, strict=True)).index((int(m.group(1)), int(m.group(2))))
            line = line.replace(m.group(0), f"-23310=2,{hs[level]},{ws[level]}")
        lines.append(line)
    if len(channels) != 4:
        raise RuntimeError("model description without input shapes (convert it with tools/convert_ncnn.py)")
    states = [(channels[f"in{i}"], hs[i - 1], ws[i - 1]) for i in range(1, 5)]
    return "\n".join(lines) + "\n", states


class RvmNet:
    """One RVM network at one input size, with its recurrent state (one camera)."""

    def __init__(self, model, width, height, vulkan=True, threads=2):
        lib = library()
        if lib is None:
            raise RuntimeError(f"{LIBRARY} not found (install ncnn)")
        self.lib, self.width, self.height = lib, width, height
        template, weights = model_files(model)
        param, states = sized_param(template.read_text(), width, height)
        sized = CACHE / "ncnn" / f"{template.stem.removesuffix('.ncnn')}_{width}x{height}.param"
        sized.parent.mkdir(parents=True, exist_ok=True)
        if not sized.is_file() or sized.read_text() != param:
            tmp = sized.with_suffix(".tmp")
            tmp.write_text(param)
            tmp.replace(sized)
        self.opt = lib.ncnn_option_create()
        lib.ncnn_option_set_use_vulkan_compute(self.opt, int(vulkan))
        lib.ncnn_option_set_num_threads(self.opt, threads)
        self.net = lib.ncnn_net_create()
        lib.ncnn_net_set_option(self.net, self.opt)
        if lib.ncnn_net_load_param(self.net, str(sized).encode()) or \
                lib.ncnn_net_load_model(self.net, str(weights).encode()):
            self.close()
            raise RuntimeError(f"cannot load {weights.name}")
        self.states = []
        for c, h, w in states:
            m = lib.ncnn_mat_create_3d(w, h, c, None)
            lib.ncnn_mat_fill_float(m, 0.0)
            self.states.append(m)

    def alpha(self, frame):
        """Alpha matte (float32, input size) of a BGR uint8 frame of any size."""
        lib = self.lib
        frame = np.ascontiguousarray(frame)
        h, w = frame.shape[:2]
        src = lib.ncnn_mat_from_pixels_resize(frame.ctypes.data_as(ctypes.c_char_p), PIXEL_BGR2RGB,
                                              w, h, frame.strides[0], self.width, self.height, None)
        lib.ncnn_mat_substract_mean_normalize(src, None, NORM)
        ex = lib.ncnn_extractor_create(self.net)
        try:
            lib.ncnn_extractor_input(ex, b"in0", src)
            for i, state in enumerate(self.states, 1):
                lib.ncnn_extractor_input(ex, f"in{i}".encode(), state)
            pha = ctypes.c_void_p()
            if lib.ncnn_extractor_extract(ex, b"out1", ctypes.byref(pha)):
                raise RuntimeError("ncnn inference failed")
            new_states = []
            for i in range(2, 6):  # out2..out5: the next recurrent state
                m = ctypes.c_void_p()
                if lib.ncnn_extractor_extract(ex, f"out{i}".encode(), ctypes.byref(m)):
                    raise RuntimeError("ncnn inference failed")
                new_states.append(m.value)
        finally:
            lib.ncnn_extractor_destroy(ex)
            lib.ncnn_mat_destroy(src)
        for m in self.states:
            lib.ncnn_mat_destroy(m)
        self.states = new_states
        ph, pw = lib.ncnn_mat_get_h(pha), lib.ncnn_mat_get_w(pha)
        data = ctypes.cast(lib.ncnn_mat_get_data(pha), ctypes.POINTER(ctypes.c_float))
        out = np.ctypeslib.as_array(data, shape=(ph * pw,)).reshape(ph, pw).copy()  # 1 channel: contiguous
        lib.ncnn_mat_destroy(pha)
        return out

    def close(self):
        lib = self.lib
        for m in getattr(self, "states", []):
            lib.ncnn_mat_destroy(m)
        self.states = []
        if getattr(self, "net", None):
            lib.ncnn_net_destroy(self.net)
            self.net = None
        if getattr(self, "opt", None):
            lib.ncnn_option_destroy(self.opt)
            self.opt = None


class GpuMatter:
    """RVM on the GPU for one stream. Loading (shader compilation) happens in a thread
    while the CPU covers the first frames.

    ``width``×``height`` is the model's input size: frames of another size are scaled to it,
    and the cut-out back to the frame's size.
    ``status``: starting | ready | error:<message>
    """

    def __init__(self, width, height, model, vulkan=True):
        self.width, self.height, self.model = width, height, model
        self.status, self.net = "starting", None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._load, args=(vulkan,), daemon=True, name="gpu-load")
        self.thread.start()

    def _load(self, vulkan):
        try:
            net = RvmNet(self.model, self.width, self.height, vulkan)
        except Exception as e:
            self.status = f"error:{e}"
            return
        with self.lock:
            if self.status == "closed":
                net.close()
                return
            self.net, self.status = net, "ready"

    def ready(self):
        return self.status == "ready"

    def alpha(self, frame):
        h, w = frame.shape[:2]
        try:
            with self.lock:
                a = self.net.alpha(frame)
        except Exception as e:
            self.status = f"error:{e}"
            return None
        if a.shape != (h, w):
            a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
        return a

    def close(self):
        with self.lock:
            self.status = "closed"
            if self.net is not None:
                self.net.close()
                self.net = None
