"""Tiny JSON-based translation layer.

Each language is a file in ``locales/<code>.json`` mapping message keys to text.
``en.json`` is the reference: missing keys fall back to English. To add a
language, copy ``en.json`` to ``<code>.json`` (e.g. ``de.json``) and translate the
values — see CONTRIBUTING.md.
"""

import json
import os
from importlib import resources

_strings: dict[str, str] = {}
_fallback: dict[str, str] = {}
current = "en"


def _load(code):
    path = resources.files(__package__).joinpath("locales", f"{code}.json")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None


def available():
    """Codes of the bundled languages, e.g. ['en', 'fr']."""
    folder = resources.files(__package__).joinpath("locales")
    return sorted(p.name[:-5] for p in folder.iterdir() if p.name.endswith(".json"))


def _candidates(preferred):
    if preferred:
        yield preferred
    for var in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
        for part in os.environ.get(var, "").split(":"):
            code = part.split(".")[0].split("@")[0]
            if code and code not in ("C", "POSIX"):
                yield code
                yield code.split("_")[0]


def setup(preferred=None):
    """Pick the language: ``preferred`` (from the config) first, then the system locale."""
    global _strings, _fallback, current
    _fallback = _load("en") or {}
    for code in _candidates(preferred):
        strings = _load(code)
        if strings is not None:
            _strings, current = strings, code
            return
    _strings, current = _fallback, "en"


def tr(key, **kwargs):
    text = _strings.get(key) or _fallback.get(key) or key
    return text.format(**kwargs) if kwargs else text


def number(value, decimals=0):
    """``value`` with the language's decimal separator."""
    return f"{value:.{decimals}f}".replace(".", tr("number.decimal"))
