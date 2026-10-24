import json
import re
from pathlib import Path

import pytest

from webcam_blur import i18n, v4l2

LOCALES = Path(__file__).resolve().parent.parent / "webcam_blur" / "locales"
REFERENCE = json.loads((LOCALES / "en.json").read_text(encoding="utf-8"))


def placeholders(text):
    return set(re.findall(r"{(\w+)}", text))


@pytest.mark.parametrize("path", sorted(LOCALES.glob("*.json")), ids=lambda p: p.stem)
def test_locale_file(path):
    strings = json.loads(path.read_text(encoding="utf-8"))
    assert strings.get("_language"), "set _language to the language's own name"
    unknown = set(strings) - set(REFERENCE)
    assert not unknown, f"keys not in en.json: {sorted(unknown)}"
    for key, text in strings.items():
        if key.startswith("_"):
            continue
        assert isinstance(text, str) and text.strip(), key
        assert placeholders(text) == placeholders(REFERENCE[key]), f"{key}: placeholders differ from en.json"
        text.format(**{p: "x" for p in placeholders(text)})  # must be a valid format string


@pytest.mark.parametrize("code", i18n.available())
def test_virtual_camera_suffix_leaves_room_for_a_name(code):
    i18n.setup(code)
    suffix = i18n.tr("virtual_camera.label", name="")
    assert len(suffix.encode()) <= 16, "virtual_camera.label is too long for v4l2loopback's 31-byte names"


def test_language_selection(monkeypatch):
    for var in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LANG", "fr_FR.UTF-8")
    i18n.setup()
    assert i18n.current == "fr"
    i18n.setup("en")  # explicit choice wins over the locale
    assert i18n.current == "en"
    monkeypatch.setenv("LANG", "xx_YY.UTF-8")  # unknown language: English
    i18n.setup()
    assert i18n.current == "en"


def test_missing_key_falls_back_to_english(monkeypatch):
    i18n.setup("fr")
    monkeypatch.setitem(i18n._strings, "menu.quit", "")
    assert i18n.tr("menu.quit") == REFERENCE["menu.quit"]


def test_labels_fit_v4l2loopback():
    for name in ("C922 Pro Stream Webcam", "Integrated Camera: Integrated C", "Caméra intégrée très très longue"):
        for template in ("{name} (blur)", "{name} (flou)"):
            label = v4l2.fit_label(template, name)
            assert len(label.encode()) <= v4l2.LABEL_MAX
            assert label.endswith(template.format(name="").strip())
    assert v4l2.fit_label("{name} (blur)", "C922 Pro Stream Webcam") == "C922 Pro Stream Webcam (blur)"
