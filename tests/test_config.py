import json

from webcam_blur.config import CAMERA_DEFAULTS, DEFAULTS, Config


def test_defaults(tmp_path):
    cfg = Config(tmp_path / "config.json")
    assert cfg["strength"] == DEFAULTS["strength"]
    assert cfg["use_gpu"] is True
    assert cfg.camera("cam") == CAMERA_DEFAULTS


def test_roundtrip(tmp_path):
    path = tmp_path / "config.json"
    cfg = Config(path)
    cfg["strength"] = "strong"
    cfg.camera("cam")["blur"] = False
    cfg.save()
    again = Config(path)
    assert again["strength"] == "strong"
    assert again.camera("cam")["blur"] is False


def test_invalid_values_are_reset(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"strength": "extreme"}))
    assert Config(path)["strength"] == DEFAULTS["strength"]
    path.write_text("{not json")
    assert Config(path)["strength"] == DEFAULTS["strength"]


def test_migrates_pre_release_format(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "strength": "fort", "mode": "vert", "quality": "haute", "green": "#00ff00",
        "cameras": [
            {"name": "C922", "input": "/dev/v4l/by-id/usb-046d_C922-video-index0", "effect": False},
            {"name": "C920", "input": "/dev/v4l/by-id/usb-046d_C920-video-index0", "blur": True},
        ],
    }))
    cfg = Config(path)
    assert cfg["strength"] == "strong"
    assert "mode" not in cfg.data and "green" not in cfg.data
    assert cfg.camera("usb-046d_C922-video-index0")["blur"] is False
    assert cfg.camera("usb-046d_C920-video-index0")["blur"] is True
