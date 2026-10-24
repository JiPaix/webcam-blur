"""Robust Video Matting model files: where they are, and downloading them if missing.

Kept free of heavy imports: the tray process uses it at startup.
"""

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

MODEL_URL = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/"
MODELS = {
    "rvm_mobilenetv3_fp32.onnx": "88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828",
    "rvm_resnet50_fp32.onnx": "25db300fcb6ee27f941a1b52c97856e8d1f13c7f35817f81a612f89af0e8a85c",
}
CPU_MODEL = "rvm_mobilenetv3_fp32.onnx"
# GPU model per quality setting. Both run in half precision on MIGraphX. "standard" needs
# about 600 MB less memory during a call; "high" has slightly finer edges.
GPU_MODELS = {"standard": "rvm_mobilenetv3_fp32.onnx", "high": "rvm_resnet50_fp32.onnx"}

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
