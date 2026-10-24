"""Starting this package's own modules as subprocesses."""

import os
import sys
from pathlib import Path


def module_command(name):
    """(argv, env) to run ``python -m webcam_blur.<name>``, also from a source checkout."""
    root = str(Path(__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (root, env.get("PYTHONPATH")) if p)
    return [sys.executable, "-m", f"webcam_blur.{name}"], env
