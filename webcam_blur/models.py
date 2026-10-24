"""Robust Video Matting model files: where they are, and downloading them if missing.

Kept free of heavy imports: the tray process uses it at startup.
"""

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

RVM_URL = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/"
# RVM converted for ncnn (GPU, Vulkan) by tools/convert_ncnn.py. Same license as RVM (GPL-3.0).
NCNN_URL = "https://github.com/JiPaix/webcam-blur/releases/download/ncnn-models-1/"
MODELS = {  # name: (download URL folder, sha256)
    "rvm_mobilenetv3_fp32.onnx": (RVM_URL, "88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828"),
    "rvm_resnet50_fp32.onnx": (RVM_URL, "25db300fcb6ee27f941a1b52c97856e8d1f13c7f35817f81a612f89af0e8a85c"),
    "rvm_mobilenetv3.ncnn.param": (NCNN_URL, "7064b3781bc7652d01e10fc5684349fb4c0df700a4841b032835fbd15275d8d5"),
    "rvm_mobilenetv3.ncnn.bin": (NCNN_URL, "f34dda8b84610850672c2c4c226c74022d9487ae85ac8806ca56952d3b00344f"),
    "rvm_resnet50.ncnn.param": (NCNN_URL, "99710e4929b5684ba09003adf7169960b48b7ee7ac5e2811888a55e8d7eed328"),
    "rvm_resnet50.ncnn.bin": (NCNN_URL, "0a0f93e0171a996f162999d4bc8360d597b638fc106ac3348e142d4cd989ec21"),
}
CPU_MODEL = "rvm_mobilenetv3_fp32.onnx"
# Model per quality setting, on the CPU (onnxruntime) and the GPU (ncnn, its .ncnn files).
# "low" and "standard" are the same model at different input sizes (see matting.CPU_WIDTH
# and GPU_WIDTH below); "high" has slightly finer edges and needs about 2.5x the computing.
QUALITY_MODELS = {"low": "rvm_mobilenetv3_fp32.onnx", "standard": "rvm_mobilenetv3_fp32.onnx",
                  "high": "rvm_resnet50_fp32.onnx"}
GPU_WIDTH = {"low": 640}  # widest model input per quality; others: the camera's full resolution


def gpu_input_size(width, height, quality):
    limit = GPU_WIDTH.get(quality, width)
    if width <= limit:
        return width, height
    return limit, round(height * limit / width / 2) * 2


USER_DATA = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "webcam-blur"
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "webcam-blur"
MODEL_DIRS = [Path(p) for p in (os.environ.get("WEBCAM_BLUR_MODEL_DIR"),) if p] + [
    Path("/usr/share/webcam-blur/models"), USER_DATA / "models"]


def model_path(name):
    for folder in MODEL_DIRS:
        if (folder / name).is_file():
            return folder / name
    return None


def ensure_models():
    """Download missing models to the user data dir (source checkouts; packages ship them)."""
    for name, (url, sha) in MODELS.items():
        if model_path(name):
            continue
        dest = USER_DATA / "models" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {name}…", file=sys.stderr, flush=True)
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url + name, tmp)
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if digest != sha:
            tmp.unlink()
            raise RuntimeError(f"{name}: checksum mismatch ({digest})")
        tmp.replace(dest)
