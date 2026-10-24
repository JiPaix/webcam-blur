"""Person matting with Robust Video Matting (RVM), on CPU or in a GPU subprocess.

- CPU: RVM mobilenetv3, frame downscaled to at most 640 px wide.
- GPU: RVM resnet50 at full resolution. GPU execution providers such as MIGraphX
  need static shapes, so the model is "frozen" once per camera resolution
  (constant downsample_ratio, fixed input sizes) and cached. Inference runs in a
  subprocess started for each stream and stopped afterwards, so nothing stays on
  the GPU while no app uses a camera.
"""

import base64
import hashlib
import multiprocessing as mp
import os
import sys
import threading
import urllib.request
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODEL_URL = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/"
MODELS = {
    "rvm_mobilenetv3_fp32.onnx": "88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828",
    "rvm_resnet50_fp32.onnx": "25db300fcb6ee27f941a1b52c97856e8d1f13c7f35817f81a612f89af0e8a85c",
}
CPU_MODEL = "rvm_mobilenetv3_fp32.onnx"
GPU_MODEL = "rvm_resnet50_fp32.onnx"

USER_DATA = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "webcam-blur"
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "webcam-blur"
MODEL_DIRS = [Path(p) for p in (os.environ.get("WEBCAM_BLUR_MODEL_DIR"),) if p] + [
    Path("/usr/share/webcam-blur/models"), USER_DATA / "models"]

# GPU execution providers, in order of preference.
GPU_PROVIDERS = ("MIGraphXExecutionProvider", "CUDAExecutionProvider", "ROCMExecutionProvider")
GPU_TIMEOUT = 2.0  # seconds to wait for one GPU frame before falling back to CPU


def model_path(name):
    for folder in MODEL_DIRS:
        if (folder / name).is_file():
            return folder / name
    return None


def ensure_models():
    """Download missing models to the user data dir (source checkouts; packages ship them)."""
    for name, sha in MODELS.items():
        if model_path(name):
            continue
        dest = USER_DATA / "models" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {name}…", file=sys.stderr, flush=True)
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL + name, tmp)
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if digest != sha:
            tmp.unlink()
            raise RuntimeError(f"{name}: checksum mismatch ({digest})")
        tmp.replace(dest)


def gpu_provider():
    available = ort.get_available_providers()
    return next((p for p in GPU_PROVIDERS if p in available), None)


# y = x + x on a [1, 16] tensor: enough to prove a GPU provider really runs here.
PROBE_MODEL = base64.b64decode(
    "CAgSC3dlYmNhbS1ibHVyOkEKDgoBeAoBeBIBeSIDQWRkEgVwcm9iZVoTCgF4Eg4KDAgBEggKAggBCgIIEGITCgF5Eg4KDAgB"
    "EggKAggBCgIIEEIECgAQDQ==")


def _probe_process(conn, provider):
    try:
        sess = ort.InferenceSession(PROBE_MODEL, providers=[provider])
        if sess.get_providers()[0] != provider:
            raise RuntimeError(f"{provider} fell back to {sess.get_providers()[0]}")
        y = sess.run(None, {"x": np.ones((1, 16), np.float32)})[0]
        if not np.allclose(y, 2.0):
            raise RuntimeError("wrong result")
        conn.send("ok")
    except Exception as e:
        conn.send(f"{type(e).__name__}: {e}")


def probe(provider, timeout=60):
    """Run the probe model on ``provider`` in a throwaway process. Returns None if it works,
    else the reason. A separate process keeps a broken driver from crashing or hanging the
    app, and leaves nothing allocated on the GPU afterwards."""
    ctx = mp.get_context("spawn")
    conn, child = ctx.Pipe()
    proc = ctx.Process(target=_probe_process, args=(child, provider), daemon=True)
    proc.start()
    child.close()
    try:
        if not conn.poll(timeout):
            return "timed out"
        result = conn.recv()
    except EOFError:
        result = f"probe process crashed (exit code {proc.exitcode})"
    finally:
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        conn.close()
    return None if result == "ok" else result


class GpuSupport:
    """Whether GPU inference works on this machine, checked once at startup.

    ``state``: checking | ok | no_onnx (GPU works, python-onnx missing) | unsupported
    """

    def __init__(self):
        self.state = "checking"
        self.provider = None
        self.detail = ""

    def check(self, on_done=None):
        def run():
            provider = gpu_provider()
            if provider is None:
                self.state, self.detail = "unsupported", "onnxruntime has no GPU execution provider"
            else:
                error = probe(provider)
                if error:
                    self.state, self.detail = "unsupported", f"{provider}: {error}"
                else:
                    self.provider = provider
                    self.state = "ok" if have_onnx() else "no_onnx"
            if self.state == "unsupported":
                print(f"GPU acceleration unavailable ({self.detail})", file=sys.stderr, flush=True)
            if on_done:
                on_done()
        threading.Thread(target=run, daemon=True, name="gpu-check").start()


gpu_support = GpuSupport()


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

    def __init__(self, session):
        self.session = session
        self.reset()

    @staticmethod
    def create_session(threads=3):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        return ort.InferenceSession(str(model_path(CPU_MODEL)), opts, providers=["CPUExecutionProvider"])

    def reset(self):
        self.rec = [np.zeros((1, 1, 1, 1), np.float32)] * 4

    def alpha(self, frame):
        h, w = frame.shape[:2]
        scale = min(1.0, 640 / w)
        small = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) \
            if scale < 1 else frame
        sh, sw = small.shape[:2]
        x = (cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        _fgr, pha, *self.rec = self.session.run(None, {
            "src": x, "r1i": self.rec[0], "r2i": self.rec[1], "r3i": self.rec[2], "r4i": self.rec[3],
            "downsample_ratio": np.array([downsample_ratio(sw, sh)], np.float32),
        })
        a = pha[0, 0]
        return cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR) if a.shape != (h, w) else a


# --- Static model for GPU providers -------------------------------------------

def frozen_model(width, height):
    """Path of the static-shape GPU model for this resolution, building it if needed."""
    dest = CACHE / "models" / f"rvm_resnet50_{width}x{height}.onnx"
    if dest.is_file():
        return dest
    import onnx
    from onnx import numpy_helper

    src = str(model_path(GPU_MODEL))
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
        # One cache folder per model: compiled files of other resolutions are left alone.
        cache = CACHE / "migraphx" / path.stem
        cache.mkdir(parents=True, exist_ok=True)
        before = set(cache.glob("*.mxr"))
        if not before:
            conn.send("compiling")
        options = {"migraphx_model_cache_dir": str(cache)}
    sess = ort.InferenceSession(str(path), providers=[(provider, options)])
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


def gpu_process(conn, frame_name, alpha_name, width, height, provider):
    """GPU subprocess: reads frames from shared memory and writes the alpha back."""
    shm_f = _attach(frame_name)
    shm_a = _attach(alpha_name)
    frame = np.ndarray((height, width, 3), np.uint8, shm_f.buf)
    alpha = np.ndarray((height, width), np.float32, shm_a.buf)
    try:
        if not (CACHE / "models" / f"rvm_resnet50_{width}x{height}.onnx").is_file():
            conn.send("preparing")
        path = frozen_model(width, height)
        sess = _gpu_session(path, provider, conn)
        shapes = {i.name: i.shape for i in sess.get_inputs()}
        rec = [np.zeros(shapes[f"r{k}i"], np.float32) for k in range(1, 5)]
        _, _, *rec = sess.run(None, {"src": np.zeros(shapes["src"], np.float32),
                                     "r1i": rec[0], "r2i": rec[1], "r3i": rec[2], "r4i": rec[3]})
        rec = [np.zeros(shapes[f"r{k}i"], np.float32) for k in range(1, 5)]
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

    ``status``: starting | preparing | compiling | ready | error:<message>
    """

    def __init__(self, width, height, provider):
        self.width, self.height = width, height
        self.shm_f = shared_memory.SharedMemory(create=True, size=width * height * 3)
        self.shm_a = shared_memory.SharedMemory(create=True, size=width * height * 4)
        self.frame = np.ndarray((height, width, 3), np.uint8, self.shm_f.buf)
        self.alpha_buf = np.ndarray((height, width), np.float32, self.shm_a.buf)
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=gpu_process, daemon=True, args=(
            child, self.shm_f.name, self.shm_a.name, width, height, provider))
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
        self.frame[:] = frame
        try:
            self.conn.send("run")
            if not self.conn.poll(GPU_TIMEOUT):
                raise TimeoutError("GPU not responding")
            self.conn.recv()
        except (OSError, EOFError, TimeoutError) as e:
            self.status = f"error:{e}"
            return None
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
