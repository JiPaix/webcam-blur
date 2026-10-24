"""User settings, stored as JSON in ~/.config/webcam-blur/config.json."""

import json
import os
from pathlib import Path

PATH = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "webcam-blur" / "config.json"

STRENGTHS = {"light": 3.0, "medium": 6.0, "strong": 12.0}  # blur sigma, in quarter-resolution pixels

DEFAULTS = {
    "strength": "medium",
    "use_gpu": True,
    "language": None,   # None = follow the system locale
    "cameras": {},      # per-camera settings, keyed by Camera.key
}

# Per camera: "blur" (effect on/off), "virtual" (create its virtual camera), and
# optionally "width"/"height"/"fps" to override the automatic capture mode.
CAMERA_DEFAULTS = {"blur": True, "virtual": True}


def _migrate(data):
    """Settings from the pre-release single-file version (cameras as a list, French keys)."""
    if isinstance(data.get("cameras"), list):
        data["cameras"] = {os.path.basename(c["input"]): {"blur": c.get("blur", c.get("effect", True))}
                           for c in data["cameras"] if "input" in c}
    old_strengths = {"leger": "light", "moyen": "medium", "fort": "strong"}
    if data.get("strength") in old_strengths:
        data["strength"] = old_strengths[data["strength"]]
    for key in ("mode", "quality", "green"):
        data.pop(key, None)
    return data


class Config:
    def __init__(self, path=PATH):
        self.path = path
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            data = {}
        self.data = {**DEFAULTS, **_migrate(data)}
        if self.data["strength"] not in STRENGTHS:
            self.data["strength"] = DEFAULTS["strength"]

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value
        self.save()

    def camera(self, key):
        """Settings dict of one camera (created with defaults on first sight)."""
        cam = self.data["cameras"].setdefault(key, {})
        for k, v in CAMERA_DEFAULTS.items():
            cam.setdefault(k, v)
        return cam

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False) + "\n")
        tmp.replace(self.path)
