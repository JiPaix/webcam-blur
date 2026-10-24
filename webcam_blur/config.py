"""User settings, stored as JSON in ~/.config/webcam-blur/config.json."""

import copy
import json
import os
from pathlib import Path

PATH = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "webcam-blur" / "config.json"

STRENGTHS = {"light": 3.0, "medium": 6.0, "strong": 12.0}  # blur sigma, in quarter-resolution pixels
QUALITIES = ("low", "standard", "high")  # see models.QUALITY_MODELS and matting.CPU_WIDTH

DEFAULTS = {
    "strength": "medium",
    "use_gpu": False,   # Low quality is as light on the CPU, without the GPU stack's memory
    "quality": "low",
    "language": None,   # None = follow the system locale
    "cameras": {},      # per-camera settings, keyed by Camera.key
}

# Per camera: "blur" (effect on/off), "virtual" (create its virtual camera), "mode"
# ({"width", "height", "fps"} chosen in the settings, or None for automatic) and "controls"
# (image controls changed in the settings, {control key: value}, set again on plug-in).
CAMERA_DEFAULTS = {"blur": True, "virtual": True, "mode": None, "controls": {}}


def _migrate(data):
    """Settings from the pre-release single-file version (cameras as a list, French keys)."""
    if isinstance(data.get("cameras"), list):
        data["cameras"] = {os.path.basename(c["input"]): {"blur": c.get("blur", c.get("effect", True))}
                           for c in data["cameras"] if "input" in c}
    old_strengths = {"leger": "light", "moyen": "medium", "fort": "strong"}
    if data.get("strength") in old_strengths:
        data["strength"] = old_strengths[data["strength"]]
    if data.get("quality") == "haute":  # pre-release French value
        data["quality"] = "high"
    for key in ("mode", "green"):
        data.pop(key, None)
    for cam in data.get("cameras", {}).values():  # 0.1.0: width/height/fps overrides
        if "width" in cam and "height" in cam:
            cam["mode"] = {"width": cam.pop("width"), "height": cam.pop("height"), "fps": cam.pop("fps", 30)}
    return data


class Config:
    def __init__(self, path=PATH):
        self.path = path
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            data = {}
        self.data = {**copy.deepcopy(DEFAULTS), **_migrate(data)}
        if self.data["strength"] not in STRENGTHS:
            self.data["strength"] = DEFAULTS["strength"]
        if self.data["quality"] not in QUALITIES:
            self.data["quality"] = DEFAULTS["quality"]

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value
        self.save()

    def camera(self, key):
        """Settings dict of one camera (created with defaults on first sight)."""
        cam = self.data["cameras"].setdefault(key, {})
        for k, v in CAMERA_DEFAULTS.items():
            if k not in cam:
                cam[k] = copy.deepcopy(v)  # not one "controls" dict shared by all cameras
        return cam

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False) + "\n")
        tmp.replace(self.path)
