"""Tray application."""

import fcntl
import os
import shlex
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QAction, QActionGroup, QColor, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from . import i18n, matting
from .config import STRENGTHS, Config
from .i18n import tr
from .manager import Manager

AUTOSTART = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "autostart" / "webcam-blur.desktop"


def webcam_icon(fg, blur_on, live):
    """Drawn webcam: blue lens when blur is on, red dot while a camera is in use."""
    pm = QPixmap(64, 64)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(fg)
    p.drawRect(QRectF(29, 44, 6, 8))  # stand
    p.drawRoundedRect(QRectF(16, 52, 32, 6), 3, 3)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(fg, 5))
    p.drawEllipse(QPointF(32, 26), 19, 19)  # body
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(61, 174, 233) if blur_on else QColor(fg.red(), fg.green(), fg.blue(), 110))
    p.drawEllipse(QPointF(32, 26), 8, 8)  # lens
    if live:
        p.setBrush(QColor(237, 21, 21))
        p.drawEllipse(QPointF(53, 9), 8, 8)
    p.end()
    return QIcon(pm)


def engine_text(engine):
    if engine.startswith("gpu_error:"):
        return tr("engine.gpu_error", error=engine[len("gpu_error:"):])
    return tr(f"engine.{engine}") if engine else "…"


class Tray:
    def __init__(self, app, manager, config):
        self.app, self.manager, self.config = app, manager, config
        fg = app.palette().windowText().color()
        self.icons = {(on, live): webcam_icon(fg, on, live) for on in (False, True) for live in (False, True)}
        self.tray = QSystemTrayIcon(self.icons[(True, False)])
        self.menu = QMenu()
        self.tray.setContextMenu(self.menu)
        self.tray.activated.connect(self.on_activated)
        self.camera_set = None
        manager.changed.connect(self.refresh)
        self.timer = QTimer(interval=3000, timeout=self.refresh)
        self.timer.start()
        self.refresh()
        self.tray.show()
        if manager.error:
            self.tray.showMessage(tr("app.name"), manager.error, QSystemTrayIcon.MessageIcon.Warning)

    def build_menu(self):
        """(Re)build the menu; needed whenever the set of webcams changes."""
        m = self.menu
        m.clear()
        self.camera_actions = []
        for key, worker in sorted(self.manager.workers.items(), key=lambda kv: kv[1].camera.name):
            act = QAction(m, checkable=True)
            act.toggled.connect(lambda on, k=key: self.set_camera(k, "blur", on))
            m.addAction(act)
            self.camera_actions.append((act, worker))
        if not self.manager.workers:
            act = m.addAction(self.manager.error or tr("menu.no_cameras"))
            act.setEnabled(False)
        m.addSeparator()

        sub = m.addMenu(tr("menu.strength"))
        group = QActionGroup(sub)
        for key in STRENGTHS:
            act = QAction(tr(f"strength.{key}"), sub, checkable=True)
            act.setChecked(self.config["strength"] == key)
            act.triggered.connect(lambda _=False, k=key: self.config.__setitem__("strength", k))
            group.addAction(act)
            sub.addAction(act)

        act = m.addAction(tr("menu.use_gpu"))
        act.setCheckable(True)
        act.setChecked(self.config["use_gpu"])
        act.toggled.connect(lambda on: self.config.__setitem__("use_gpu", on))

        m.addSeparator()
        act = m.addAction(tr("menu.autostart"))
        act.setCheckable(True)
        act.setChecked(AUTOSTART.exists())
        act.toggled.connect(set_autostart)
        m.addAction(tr("menu.quit"), self.app.quit)

    def set_camera(self, key, setting, on):
        self.config.camera(key)[setting] = on
        self.config.save()
        self.refresh()

    def on_activated(self, reason):
        # Left click: blur on/off for all cameras at once.
        if reason == QSystemTrayIcon.ActivationReason.Trigger and self.camera_actions:
            any_on = any(w.settings["blur"] for _a, w in self.camera_actions)
            for act, _w in self.camera_actions:
                act.setChecked(not any_on)

    def refresh(self):
        camera_set = tuple(sorted(self.manager.workers))
        if camera_set != self.camera_set:
            self.camera_set = camera_set
            self.build_menu()
        lines = []
        for act, w in self.camera_actions:
            on = w.settings["blur"]
            if w.error:
                status = tr("status.error", error=tr(w.error))
            elif w.in_use:
                status = tr("status.in_use", engine=engine_text(w.engine))
            else:
                status = tr("status.idle")
            act.blockSignals(True)
            act.setChecked(on)
            act.blockSignals(False)
            act.setText(tr("menu.camera", name=w.camera.name, status=status))
            lines.append(tr("tooltip.camera", name=w.camera.name,
                            state=tr("tooltip.on" if on else "tooltip.off"), status=status))
        if not lines:
            lines.append(self.manager.error or tr("menu.no_cameras"))
        any_on = any(w.settings["blur"] for _a, w in self.camera_actions)
        live = any(w.in_use for _a, w in self.camera_actions)
        self.tray.setIcon(self.icons[(any_on, live)])
        self.tray.setToolTip(tr("app.tooltip_title") + "\n" + "\n".join(lines))


def launch_command():
    """Command line that starts this app, for the autostart entry."""
    exe = shutil.which("webcam-blur")
    if exe:
        return exe
    root = Path(__file__).resolve().parent.parent  # running from a source checkout
    return f"env PYTHONPATH={shlex.quote(str(root))} {shlex.quote(sys.executable)} -m webcam_blur"


def set_autostart(on):
    if on:
        AUTOSTART.parent.mkdir(parents=True, exist_ok=True)
        AUTOSTART.write_text(
            "[Desktop Entry]\nType=Application\n"
            f"Name={tr('app.name')}\nExec={launch_command()}\nIcon=camera-web\n"
            "Terminal=false\nX-GNOME-Autostart-enabled=true\n")
    else:
        AUTOSTART.unlink(missing_ok=True)


def single_instance():
    """Hold a lock for the app's lifetime; return None if another instance has it."""
    path = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "webcam-blur.lock"
    f = open(path, "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return None
    return f


def log_exception(exc_type, exc, tb):
    # PyQt aborts the whole app on an exception raised in a slot unless a hook is set.
    traceback.print_exception(exc_type, exc, tb)


def main():
    sys.excepthook = log_exception
    lock = single_instance()
    if lock is None:
        print("webcam-blur is already running.", file=sys.stderr)
        return 0
    config = Config()
    i18n.setup(config["language"])
    matting.ensure_models()

    app = QApplication(sys.argv)
    app.setApplicationName("webcam-blur")
    app.setApplicationDisplayName(tr("app.name"))
    app.setDesktopFileName("webcam-blur")
    app.setQuitOnLastWindowClosed(False)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    signal.signal(signal.SIGTERM, lambda *_: app.quit())

    # Right after login the panel may not be ready yet.
    for _ in range(60):
        if QSystemTrayIcon.isSystemTrayAvailable():
            break
        time.sleep(0.5)

    manager = Manager(config)
    tray = Tray(app, manager, config)
    manager.start()
    # Let Python handle SIGINT/SIGTERM while Qt's event loop runs.
    tick = QTimer(interval=500, timeout=lambda: None)
    tick.start()
    rc = app.exec()
    manager.shutdown()
    del tray, lock
    return rc
