"""Person matting with Robust Video Matting (RVM), on CPU or in a GPU subprocess.

- CPU: RVM mobilenetv3, frame downscaled to at most 320 px ("low" quality) or 640 px wide.
- GPU: RVM mobilenetv3 at most 640 px wide ("low") or at full resolution ("standard"),
  or resnet50 at full resolution ("high"), in half precision on MIGraphX. GPU execution providers such as MIGraphX
  need static shapes, so the model is "frozen" once per camera resolution
  (constant downsample_ratio, fixed input sizes) and cached. Inference runs in a
  subprocess started for each stream and stopped afterwards, so nothing stays on
  the GPU while no app uses a camera.
"""

import base64
import ctypes
import multiprocessing as mp
import os
import shutil
from multiprocessing import shared_memory

import cv2
import numpy as np
import onnxruntime as ort

from .models import CACHE, CPU_MODEL, GPU_MODELS, MODELS, model_path  # noqa: F401  (re-exported)

# GPU execution providers, in order of preference.
GPU_PROVIDERS = ("MIGraphXExecutionProvider", "CUDAExecutionProvider", "ROCMExecutionProvider")
GPU_TIMEOUT = 2.0  # seconds to wait for one GPU frame before falling back to CPU

# Widest frame the model sees, per quality. The cut-out is scaled back up: edges barely
# change, and a smaller input is much cheaper.
CPU_WIDTH = {"low": 320, "standard": 640, "high": 640}  # "high" needs the GPU
GPU_WIDTH = {"low": 640}  # other qualities: the camera's full resolution

# OpenCV's default (one thread per core) spends more time handing out work and waiting
# than computing on frames this size.
CV_THREADS = 2


def gpu_input_size(width, height, quality):
    limit = GPU_WIDTH.get(quality, width)
    if width <= limit:
        return width, height
    return limit, round(height * limit / width / 2) * 2


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


def gpu_provider():
    available = ort.get_available_providers()
    return next((p for p in GPU_PROVIDERS if p in available), None)


# y = x + x on a [1, 16] tensor: enough to prove a GPU provider really runs here.
PROBE_MODEL = base64.b64decode(
    "CAgSC3dlYmNhbS1ibHVyOkEKDgoBeAoBeBIBeSIDQWRkEgVwcm9iZVoTCgF4Eg4KDAgBEggKAggBCgIIEGITCgF5Eg4KDAgB"
    "EggKAggBCgIIEEIECgAQDQ==")


def try_provider(provider):
    """Run the probe model on ``provider``. Returns None if it works, else the reason."""
    try:
        sess = ort.InferenceSession(PROBE_MODEL, providers=[provider])
        if sess.get_providers()[0] != provider:
            raise RuntimeError(f"{provider} fell back to {sess.get_providers()[0]}")
        y = sess.run(None, {"x": np.ones((1, 16), np.float32)})[0]
        if not np.allclose(y, 2.0):
            raise RuntimeError("wrong result")
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def detect_gpu():
    """{"state": ok | no_onnx | unsupported, "provider", "detail"}. Meant to run in a throwaway
    process (see gpucheck): a broken driver cannot crash the app, and nothing stays on the GPU."""
    provider = gpu_provider()
    if provider is None:
        return {"state": "unsupported", "provider": None, "detail": "onnxruntime has no GPU execution provider"}
    error = try_provider(provider)
    if error:
        return {"state": "unsupported", "provider": None, "detail": f"{provider}: {error}"}
    return {"state": "ok" if have_onnx() else "no_onnx", "provider": provider, "detail": ""}


def have_onnx():
    try:
        import onnx  # noqa: F401
        return True
    except ImportError:
        return False


def downsample_ratio(width, height, target=320):
    """RVM runs its recurrent part at about ``target`` px on the long side."""
    return min(1.0, target / max(width, height))


# --- CPU ----------------------------------------------------------------------

class CpuMatter:
    """RVM mobilenetv3 on CPU; the recurrent state belongs to one camera."""

    def __init__(self, session, width=CPU_WIDTH["standard"]):
        self.session = session
        self.width = width
        self.reset()

    @staticmethod
    def create_session(threads=None):
        # Without spinning, extra threads cost little and keep 60 fps within each frame's time.
        threads = threads or max(1, min(6, (os.cpu_count() or 2) // 2))
        return ort.InferenceSession(str(model_path(CPU_MODEL)), session_options(threads),
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


# --- Static model for GPU providers -------------------------------------------

def frozen_path(model, width, height):
    return CACHE / "models" / f"{model.removesuffix('_fp32.onnx')}_{width}x{height}.onnx"


def frozen_model(model, width, height):
    """Path of the static-shape version of ``model`` for this resolution, building it if needed."""
    dest = frozen_path(model, width, height)
    if dest.is_file():
        return dest
    import onnx
    from onnx import numpy_helper

    src = str(model_path(model))
    ratio = downsample_ratio(width, height)
    # Run once on CPU to learn the recurrent state shapes at this resolution.
    probe = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
    rec = [np.zeros((1, 1, 1, 1), np.float32)] * 4
    _, _, *rec = probe.run(None, {"src": np.zeros((1, 3, height, width), np.float32),
                                  "r1i": rec[0], "r2i": rec[1], "r3i": rec[2], "r4i": rec[3],
                                  "downsample_ratio": np.array([ratio], np.float32)})
    del probe
    shapes = {"src": [1, 3, height, width], **{f"r{i + 1}i": list(r.shape) for i, r in enumerate(rec)}}

    model = onnx.load(src)
    graph = model.graph
    for inp in list(graph.input):
        if inp.name == "downsample_ratio":
            graph.input.remove(inp)
            graph.initializer.append(numpy_helper.from_array(np.array([ratio], np.float32), inp.name))
        elif inp.name in shapes:
            for dim, value in zip(inp.type.tensor_type.shape.dim, shapes[inp.name], strict=True):
                dim.Clear()
                dim.dim_value = value
    for out in graph.output:
        out.type.tensor_type.shape.Clear()

    # Constant-fold the shape computations away with onnxruntime's basic optimizations.
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.onnx")
    onnx.save(model, str(tmp))
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.optimized_model_filepath = str(dest.with_suffix(".part"))
    ort.InferenceSession(str(tmp), opts, providers=["CPUExecutionProvider"])
    tmp.unlink()
    dest.with_suffix(".part").replace(dest)
    return dest


def _gpu_session(path, provider, conn):
    options = {}
    if provider == "MIGraphXExecutionProvider":
        # Half precision: same results here, less memory, faster. One cache folder per
        # model and resolution: compiled files of the others are left alone.
        cache = CACHE / "migraphx" / f"{path.stem}-fp16"
        shutil.rmtree(CACHE / "migraphx" / path.stem, ignore_errors=True)  # 0.1.0's fp32 cache
        cache.mkdir(parents=True, exist_ok=True)
        before = set(cache.glob("*.mxr"))
        if not before:
            conn.send("compiling")
        options = {"migraphx_model_cache_dir": str(cache), "migraphx_fp16_enable": "1"}
    sess = ort.InferenceSession(str(path), session_options(1), providers=[(provider, options)])
    if sess.get_providers()[0] != provider:
        raise RuntimeError(f"{provider} unavailable")
    if provider == "MIGraphXExecutionProvider":
        stale = before if set(cache.glob("*.mxr")) - before else set()
        for f in stale:  # recompiled after a ROCm/MIGraphX update: old files are useless
            f.unlink(missing_ok=True)
    return sess


def _attach(name):
    """Attach to the parent's shared memory; the parent owns it and unlinks it."""
    try:
        return shared_memory.SharedMemory(name=name, track=False)  # Python 3.13+
    except TypeError:
        # Older Pythons: spawned children share the parent's resource tracker, so the
        # duplicate registration is harmless (the tracker lives as long as the parent).
        return shared_memory.SharedMemory(name=name)


def gpu_process(conn, frame_name, alpha_name, width, height, provider, model):
    """GPU subprocess: reads frames from shared memory and writes the alpha back."""
    cv2.setNumThreads(CV_THREADS)
    shm_f = _attach(frame_name)
    shm_a = _attach(alpha_name)
    frame = np.ndarray((height, width, 3), np.uint8, shm_f.buf)
    alpha = np.ndarray((height, width), np.float32, shm_a.buf)
    try:
        if not frozen_path(model, width, height).is_file():
            conn.send("preparing")
        path = frozen_model(model, width, height)
        sess = _gpu_session(path, provider, conn)
        shapes = {i.name: i.shape for i in sess.get_inputs()}
        rec = [np.zeros(shapes[f"r{k}i"], np.float32) for k in range(1, 5)]
        _, _, *rec = sess.run(None, {"src": np.zeros(shapes["src"], np.float32),
                                     "r1i": rec[0], "r2i": rec[1], "r3i": rec[2], "r4i": rec[3]})
        rec = [np.zeros(shapes[f"r{k}i"], np.float32) for k in range(1, 5)]
        release_memory()  # ~250 MB left over from loading the compiled model
    except Exception as e:
        conn.send(f"error:{e}")
        return
    try:
        conn.send("ready")
    except OSError:
        return  # stream ended during preparation; the compiled model is cached for next time
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg != "run":
            return
        x = (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        _fgr, pha, *rec = sess.run(None, {"src": x, "r1i": rec[0], "r2i": rec[1], "r3i": rec[2], "r4i": rec[3]})
        alpha[:] = pha[0, 0]
        conn.send("ok")


class GpuMatter:
    """Drives one camera's GPU subprocess for the duration of a stream.

    ``width``×``height`` is the model's input size: frames of another size are scaled to it,
    and the cut-out back to the frame's size.
    ``status``: starting | preparing | compiling | ready | error:<message>
    """

    def __init__(self, width, height, provider, model=GPU_MODELS["standard"]):
        self.width, self.height, self.model = width, height, model
        self.shm_f = shared_memory.SharedMemory(create=True, size=width * height * 3)
        self.shm_a = shared_memory.SharedMemory(create=True, size=width * height * 4)
        self.frame = np.ndarray((height, width, 3), np.uint8, self.shm_f.buf)
        self.alpha_buf = np.ndarray((height, width), np.float32, self.shm_a.buf)
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=gpu_process, daemon=True, args=(
            child, self.shm_f.name, self.shm_a.name, width, height, provider, model))
        self.proc.start()
        child.close()
        self.status = "starting"

    def ready(self):
        while self.status not in ("ready",) and not self.status.startswith("error") and self.conn.poll():
            try:
                self.status = self.conn.recv()
            except EOFError:
                self.status = "error:GPU process exited"
        return self.status == "ready"

    def alpha(self, frame):
        h, w = frame.shape[:2]
        scaled = (w, h) != (self.width, self.height)
        self.frame[:] = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA) \
            if scaled else frame
        try:
            self.conn.send("run")
            if not self.conn.poll(GPU_TIMEOUT):
                raise TimeoutError("GPU not responding")
            self.conn.recv()
        except (OSError, EOFError, TimeoutError) as e:
            self.status = f"error:{e}"
            return None
        if scaled:
            return cv2.resize(self.alpha_buf, (w, h), interpolation=cv2.INTER_LINEAR)
        return self.alpha_buf.copy()

    def close(self):
        self.ready()  # pick up a late status change
        try:
            self.conn.send("quit")
        except OSError:
            pass
        if self.status in ("preparing", "compiling"):
            # Let the one-time model preparation/compilation finish in the background
            # (it exits on its own after reading "quit"); killing it would waste it.
            pass
        else:
            self.proc.join(timeout=2)
            if self.proc.is_alive():
                self.proc.kill()
                self.proc.join()
        self.conn.close()
        del self.frame, self.alpha_buf
        for shm in (self.shm_f, self.shm_a):
            shm.close()
            shm.unlink()
