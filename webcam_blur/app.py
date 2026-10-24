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

from . import gpucheck, i18n, models, v4l2
from .config import QUALITIES, STRENGTHS, Config
from .i18n import tr
from .manager import Manager
from .prepare import Precompiler, jobs_for
from .resources import ResourceMeter

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
        self.layout = None
        self.cameras = []
        manager.changed.connect(self.refresh)
        self.timer = QTimer(interval=3000, timeout=self.refresh)
        self.timer.start()
        self.meter = ResourceMeter()
        self.usage = None
        self.resources = None
        self.resource_timer = QTimer(interval=2000, timeout=self.update_resources)
        self.resource_timer.start()
        self.precompiler = Precompiler()
        self.refresh()
        self.tray.show()
        if manager.error:
            self.tray.showMessage(tr("app.name"), manager.error, QSystemTrayIcon.MessageIcon.Warning)

    def build_menu(self):
        """(Re)build the menu; needed whenever the webcams or their mode lists change."""
        m = self.menu
        m.clear()
        self.cameras = []  # (worker, submenu, blur action, {mode key or None: action}, pending note)
        for key, worker in sorted(self.manager.workers.items(), key=lambda kv: kv[1].camera.name):
            sub = m.addMenu(worker.camera.name)
            blur = sub.addAction(tr("menu.blur"))
            blur.setCheckable(True)
            blur.toggled.connect(lambda on, k=key: self.set_camera(k, "blur", on))
            res = sub.addMenu(tr("menu.resolution"))
            group = QActionGroup(res)
            choices = {}
            for mode in [None] + v4l2.menu_modes(worker.modes):
                act = res.addAction("")
                act.setCheckable(True)
                act.triggered.connect(lambda _=False, k=key, md=mode: self.set_camera(
                    k, "mode", md.to_config() if md else None))
                group.addAction(act)
                choices[mode.key() if mode else None] = act
                if mode is None:
                    res.addSeparator()
            pending = sub.addAction(tr("menu.resolution_pending"))
            pending.setEnabled(False)
            self.cameras.append((worker, sub, blur, choices, pending))
        if not self.manager.workers:
            m.addAction(self.manager.error or tr("menu.no_cameras")).setEnabled(False)
        m.addSeparator()

        sub = m.addMenu(tr("menu.strength"))
        group = QActionGroup(sub)
        for key in STRENGTHS:
            act = QAction(tr(f"strength.{key}"), sub, checkable=True)
            act.setChecked(self.config["strength"] == key)
            act.triggered.connect(lambda _=False, k=key: self.config.__setitem__("strength", k))
            group.addAction(act)
            sub.addAction(act)

        # GPU options are only offered where the startup check ran a model on the GPU.
        gpu_ok = gpucheck.gpu_support.state == "ok"
        sub = m.addMenu(tr("menu.quality"))
        group = QActionGroup(sub)
        for key in QUALITIES:
            if key == "high" and not gpu_ok:
                continue  # far too slow on a processor
            act = QAction(tr(f"quality.{key}"), sub, checkable=True)
            act.setChecked(self.config["quality"] == key)
            act.triggered.connect(lambda _=False, k=key: self.config.__setitem__("quality", k))
            group.addAction(act)
            sub.addAction(act)
        if gpu_ok:
            act = m.addAction(tr("menu.use_gpu"))
            act.setCheckable(True)
            act.setChecked(self.config["use_gpu"])
            act.toggled.connect(lambda on: self.config.__setitem__("use_gpu", on))
        elif gpucheck.gpu_support.state == "no_onnx":
            m.addAction(tr("menu.use_gpu_needs_onnx")).setEnabled(False)

        m.addSeparator()
        act = m.addAction(tr("menu.autostart"))
        act.setCheckable(True)
        act.setChecked(AUTOSTART.exists())
        act.toggled.connect(set_autostart)
        self.resources = m.addAction("")
        self.resources.setEnabled(False)
        self.update_resources(sample=False)
        m.addAction(tr("menu.quit"), self.app.quit)

    def set_camera(self, key, setting, value):
        self.config.camera(key)[setting] = value
        self.config.save()
        self.refresh()

    def on_activated(self, reason):
        # Left click: blur on/off for all cameras at once.
        if reason == QSystemTrayIcon.ActivationReason.Trigger and self.cameras:
            any_on = any(w.settings["blur"] for w, *_ in self.cameras)
            for _w, _sub, blur, _choices, _pending in self.cameras:
                blur.setChecked(not any_on)

    def refresh(self):
        layout = (tuple((k, len(w.modes)) for k, w in sorted(self.manager.workers.items())),
                  gpucheck.gpu_support.state)
        if layout != self.layout:
            self.layout = layout
            self.build_menu()
        lines = []
        for w, sub, blur, choices, pending in self.cameras:
            on = w.settings["blur"]
            if w.error:
                status = tr("status.error", error=tr(w.error))
            elif w.in_use:
                status = tr("status.in_use", engine=engine_text(w.engine))
            else:
                status = tr("status.idle")
            state = tr("tooltip.on" if on else "tooltip.off")
            sub.setTitle(tr("menu.camera", name=w.camera.name, state=state, status=status))
            blur.blockSignals(True)
            blur.setChecked(on)
            blur.blockSignals(False)
            if w.modes:
                chosen = v4l2.find_mode(w.modes, w.settings.get("mode"))
                choices[None].setText(tr("menu.resolution_auto", mode=mode_text(v4l2.auto_mode(w.modes))))
                for mode in v4l2.menu_modes(w.modes):
                    choices[mode.key()].setText(mode_text(mode))
                selected = choices.get(chosen.key()) if chosen else None
                (selected or choices[None]).setChecked(True)
            pending.setVisible(w.mode_pending)
            lines.append(tr("tooltip.camera", name=w.camera.name, state=state, status=status))
        if not lines:
            lines.append(self.manager.error or tr("menu.no_cameras"))
        any_on = any(w.settings["blur"] for w, *_ in self.cameras)
        live = any(w.in_use for w, *_ in self.cameras)
        self.tray.setIcon(self.icons[(any_on, live)])
        self.tray.setToolTip(tr("app.tooltip_title") + "\n" + "\n".join(lines))
        self.prepare_models()

    def prepare_models(self):
        """Compile GPU models ahead of time for each webcam's resolution, so that the GPU is
        ready when an app opens a camera, whatever the quality (see prepare.py)."""
        support = gpucheck.gpu_support
        if support.state == "ok" and self.config["use_gpu"]:
            modes = [w.wanted_mode() for w in self.manager.workers.values() if w.modes]
            self.precompiler.update(jobs_for(modes, self.config["quality"], support.provider))


    def update_resources(self, sample=True):
        if sample or self.usage is None:
            self.usage = self.meter.sample()
        if self.resources is not None:
            self.resources.setText(resources_text(self.usage))


def number(value, decimals):
    return f"{value:.{decimals}f}".replace(".", tr("number.decimal"))


def size_text(n):
    if n >= 1e9:
        return tr("unit.gb", value=number(n / 1e9, 1))
    return tr("unit.mb", value=number(n / 1e6, 0))


def resources_text(usage):
    cpu = number(usage["cpu"], 1 if usage["cpu"] < 10 else 0)
    if usage["vram"] is None:
        return tr("menu.resources", cpu=cpu, ram=size_text(usage["ram"]))
    return tr("menu.resources_vram", cpu=cpu, ram=size_text(usage["ram"]), vram=size_text(usage["vram"]))


def mode_text(mode):
    return tr("mode.label", width=mode.width, height=mode.height, fps=f"{mode.fps:g}")


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
    models.ensure_models()

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
    gpucheck.gpu_support.check(on_done=manager.changed.emit)
    # Let Python handle SIGINT/SIGTERM while Qt's event loop runs.
    tick = QTimer(interval=500, timeout=lambda: None)
    tick.start()
    rc = app.exec()
    manager.shutdown()
    del tray, lock
    return rc
