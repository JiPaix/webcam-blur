"""Keeps one virtual camera per plugged-in webcam."""

import sys

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from . import cameras, v4l2
from .i18n import tr
from .worker import CameraWorker


class Manager(QObject):
    """Matches virtual cameras to plugged-in webcams. All methods run on the Qt thread."""

    changed = pyqtSignal()
    hotplug = pyqtSignal()

    def __init__(self, config, scan=cameras.scan):
        super().__init__()
        self.config = config
        self.scan = scan  # tests pass a fake camera list
        self.workers = {}   # camera key -> CameraWorker
        self.orphans = {}   # camera key -> device nr still open by an app, to remove later
        self.cameras = {}   # camera key -> Camera currently plugged in
        self.skipped = set()  # keys of capture devices without colour formats (IR cameras…)
        self.error = None
        try:
            self.control = v4l2.LoopbackControl()
        except OSError as e:
            self.control = None
            self.error = tr("error.no_control", error=e.strerror or e)
        # udev sends several events per plug/unplug: scan once they settle.
        self.debounce = QTimer(singleShot=True, interval=500, timeout=self.sync)
        self.hotplug.connect(self.debounce.start)
        self.monitor = cameras.HotplugMonitor(self.hotplug.emit)
        # Retries removals of virtual cameras that were still open when their webcam left.
        self.timer = QTimer(interval=5000, timeout=self.sync)
        self.started = False

    def label(self, camera):
        return v4l2.fit_label(tr("virtual_camera.label"), camera.name)

    def start(self):
        if self.control:
            # Virtual cameras left over from a previous run (e.g. still open by an app).
            for nr, label in self.control.devices().items():
                for cam in self.scan().values():
                    if label == self.label(cam):
                        self.orphans[cam.key] = nr
        self.monitor.start()
        self.timer.start()
        self.started = True
        self.sync()

    def sync(self):
        self.cameras = self.scan()
        if self.control:
            for key, cam in self.cameras.items():
                if key not in self.workers and self.config.camera(key)["virtual"]:
                    self.add(cam)
            for key in list(self.workers):
                if key not in self.cameras or not self.config.camera(key)["virtual"]:
                    self.drop(key)
            for key, nr in list(self.orphans.items()):
                if key not in self.workers and self.control.remove(nr):
                    del self.orphans[key]
        self.changed.emit()

    def add(self, cam):
        if cam.key in self.skipped:
            return
        try:
            modes = v4l2.list_modes(cam.path)
        except OSError:
            return  # not ready yet; next sync retries
        if not modes:
            self.skipped.add(cam.key)  # e.g. infrared face-login cameras (greyscale only)
            return
        saved = self.config.camera(cam.key)["controls"]
        if saved:  # webcams forget their image settings when unplugged
            try:
                failed = v4l2.apply_controls(cam.path, saved)
            except OSError:
                failed = list(saved)
            if failed:
                print(f"[{cam.name}] cannot restore {', '.join(failed)}", file=sys.stderr, flush=True)
        try:
            nr = self.orphans.pop(cam.key, None)
            if nr is None:
                nr = self.control.add(self.label(cam))
        except OSError as e:
            print(f"[{cam.name}] cannot create virtual camera: {e}", file=sys.stderr, flush=True)
            return
        worker = CameraWorker(cam, nr, self.config, self.changed.emit)
        self.workers[cam.key] = worker
        worker.start()

    def drop(self, key):
        worker = self.workers.pop(key)
        worker.stop()
        if not self.control.remove(worker.device_nr):
            self.orphans[key] = worker.device_nr

    def shutdown(self):
        if self.started:
            self.monitor.stop()
            self.timer.stop()
            self.started = False
        for key in list(self.workers):
            self.drop(key)
