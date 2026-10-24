"""Convert RVM's ONNX models to ncnn for the Vulkan GPU path (developers only).

The result is published as a GitHub release asset (see CONTRIBUTING.md), and its
checksums go into webcam_blur/models.py. Needs: onnx, onnxruntime, pnnx (pip install pnnx),
ncnn (the library, for the check) and the RVM ONNX models (downloaded if missing).

    python tools/convert_ncnn.py [output folder]

Each model is frozen at 640x360 (static shapes, as pnnx needs), converted with pnnx in
half precision, then patched:
- the channel mean of the guided filter (a Reduction layer) is computed with a slice, two
  additions and a scale instead: ncnn's Vulkan Reduction gives wrong results on some GPUs
  (seen on RADV, RDNA 4), which shows as stripes along the edges of the cut-out;
- the Input layers get their template shapes (0=w 1=h 2=c): the app reads the recurrent
  states' channel counts from there (they differ between models).
The converted network is then checked against onnxruntime at two sizes, the second one
generated from the 640x360 template like the app does.
"""

import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from webcam_blur import models, ncnn  # noqa: E402

TEMPLATE = (640, 360)
PORTRAIT = Path(__file__).resolve().parent.parent / "tests" / "data" / "portrait.jpg"


def downsample_ratio(width, height):
    return min(1.0, ncnn.TARGET / max(width, height))


def freeze(src, dest, width, height):
    """Static-shape copy of an RVM model for one size (downsample_ratio becomes a constant)."""
    import onnx
    from onnx import numpy_helper

    ratio = downsample_ratio(width, height)
    probe = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    rec = [np.zeros((1, 1, 1, 1), np.float32)] * 4
    _, _, *rec = probe.run(None, {"src": np.zeros((1, 3, height, width), np.float32),
                                  **{f"r{i}i": rec[i - 1] for i in range(1, 5)},
                                  "downsample_ratio": np.array([ratio], np.float32)})
    shapes = {"src": [1, 3, height, width], **{f"r{i + 1}i": list(r.shape) for i, r in enumerate(rec)}}
    model = onnx.load(str(src))
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
    tmp = dest.with_suffix(".tmp.onnx")
    onnx.save(model, str(tmp))
    # Constant-fold the shape computations away with onnxruntime's basic optimizations.
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.optimized_model_filepath = str(dest)
    ort.InferenceSession(str(tmp), opts, providers=["CPUExecutionProvider"])
    tmp.unlink()


def fix_channel_mean(param):
    """Replace RVM's x.mean(1, keepdim=True) Reduction layers by slice + adds + scale."""
    lines = param.splitlines()
    layers, blobs = map(int, lines[1].split())
    out, n = [], 0
    for line in lines[2:]:
        f = line.split()
        if f and f[0] == "Reduction" and "0=3" in f and "-23303=1,0" in f and "4=1" in f:
            name, x, y = f[1], f[4], f[5]
            out += [f"Slice {name}_ch 1 3 {x} {name}_r {name}_g {name}_b -23300=3,1,1,1 1=0",
                    f"BinaryOp {name}_rg 2 1 {name}_r {name}_g {name}_s1 0=0",
                    f"BinaryOp {name}_rgb 2 1 {name}_s1 {name}_b {name}_s2 0=0",
                    f"BinaryOp {name}_third 1 1 {name}_s2 {y} 0=2 1=1 2=3.333333e-01"]
            n += 1
        else:
            out.append(line)
    if not n:
        raise SystemExit("no channel-mean Reduction found: the conversion changed, check the model")
    return "\n".join([lines[0], f"{layers + 3 * n} {blobs + 5 * n}"] + out) + "\n", n


def add_input_shapes(param, frozen):
    """Write each input's shape (as at 640x360) into its Input layer."""
    shapes = {f"in{i}": inp.shape for i, inp in enumerate(
        ort.InferenceSession(str(frozen), providers=["CPUExecutionProvider"]).get_inputs())}
    out = []
    for line in param.splitlines():
        f = line.split()
        if f and f[0] == "Input" and f[1] in shapes:
            _, c, h, w = shapes[f[1]]
            line = f"{line} 0={w} 1={h} 2={c}"
        out.append(line)
    return "\n".join(out) + "\n"


def reference_alpha(src, frame, steps):
    h, w = frame.shape[:2]
    sess = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    x = (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255).transpose(2, 0, 1)[None].copy()
    rec = [np.zeros((1, 1, 1, 1), np.float32)] * 4
    for _ in range(steps):
        _, pha, *rec = sess.run(None, {"src": x, **{f"r{i}i": rec[i - 1] for i in range(1, 5)},
                                       "downsample_ratio": np.array([downsample_ratio(w, h)], np.float32)})
    return pha[0, 0]


def check(model, src, vulkan):
    portrait = cv2.imread(str(PORTRAIT))
    for width, height in (TEMPLATE, (640, 480)):
        frame = cv2.resize(portrait, (width, height))
        net = ncnn.RvmNet(model, width, height, vulkan=vulkan)
        for _ in range(5):
            a = net.alpha(frame)
        net.close()
        ref = reference_alpha(src, frame, 5)
        agreement = ((a > 0.5) == (ref > 0.5)).mean() * 100
        diff = float(np.abs(a - ref).mean())
        where = "Vulkan" if vulkan else "CPU"
        print(f"  {where:6s} {width}x{height}: agreement {agreement:.3f} %, mean |diff| {diff:.5f}")
        if agreement < 99.9 or diff > 0.002:
            raise SystemExit(f"{model} on ncnn ({where}) does not match onnxruntime")


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "dist/ncnn-models").resolve()
    out.mkdir(parents=True, exist_ok=True)
    pnnx = shutil.which("pnnx")
    if pnnx is None:
        raise SystemExit("pnnx not found: pip install pnnx")
    models.ensure_models()
    for model in dict.fromkeys(models.QUALITY_MODELS.values()):
        stem = model.removesuffix("_fp32.onnx")
        src = models.model_path(model)
        print(f"{stem}:")
        with tempfile.TemporaryDirectory() as tmp:
            frozen = Path(tmp) / "rvm.onnx"
            freeze(src, frozen, *TEMPLATE)
            subprocess.run([pnnx, str(frozen), "fp16=1"], cwd=tmp, check=True, capture_output=True)
            param, n = fix_channel_mean((Path(tmp) / "rvm.ncnn.param").read_text())
            param = add_input_shapes(param, frozen)
            (out / f"{stem}.ncnn.param").write_text(param)
            shutil.copy(Path(tmp) / "rvm.ncnn.bin", out / f"{stem}.ncnn.bin")
        print(f"  converted, {n} channel-mean layers patched")
        models.MODEL_DIRS.insert(0, out)  # check the files just written
        check(model, src, vulkan=False)
        if ncnn.gpu_count():
            check(model, src, vulkan=True)
        models.MODEL_DIRS.pop(0)
    print("\nchecksums for webcam_blur/models.py:")
    for f in sorted(out.glob("*.ncnn.*")):
        print(f'    "{f.name}": "{hashlib.sha256(f.read_bytes()).hexdigest()}",')


if __name__ == "__main__":
    main()
