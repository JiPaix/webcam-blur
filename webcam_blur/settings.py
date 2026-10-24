"""Webcam settings window: one tab per webcam with blur, resolution and the webcam's own
image controls (exposure, white balance, focus…).

The image controls live in the webcam and also apply to the raw camera. The ones changed
here are saved in the config and set again whenever the webcam is plugged in (see
Manager.add), since webcams forget them when unplugged.
"""

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                             QPushButton, QScrollArea, QSlider, QTabWidget, QVBoxLayout, QWidget)

from . import v4l2
from .i18n import number, tr

# Known controls, in display order, by section. Others follow, under "Other", as the
# webcam lists them.
SECTIONS = {
    "image": ["brightness", "contrast", "saturation", "hue", "sharpness", "gamma", "gain",
              "backlight_compensation"],
    "white_balance": ["white_balance_automatic", "white_balance_temperature"],
    "exposure": ["auto_exposure", "exposure_time_absolute", "exposure_dynamic_framerate",
                 "power_line_frequency"],
    "focus": ["focus_automatic_continuous", "focus_absolute", "zoom_absolute", "pan_absolute",
              "tilt_absolute"],
}
EXPOSURE = "exposure_time_absolute"  # in units of 100 µs


def mode_text(mode):
    return tr("mode.label", width=mode.width, height=mode.height, fps=f"{mode.fps:g}")


def control_name(control):
    key = f"control.{control.key}"
    name = tr(key)
    return control.name if name == key else name


def menu_item_name(control, index):
    key = f"control.{control.key}.{index}"
    name = tr(key)
    return control.menu[index] if name == key else name


def value_text(control, value):
    if control.key == EXPOSURE:
        return tr("unit.ms", value=number(value / 10, 1))
    if control.key == "white_balance_temperature":
        return tr("unit.kelvin", value=value)
    if control.key in ("pan_absolute", "tilt_absolute"):  # arc seconds
        return f"{number(value / 3600)}°"
    return str(value)


def sections(controls):
    known = {key: name for name, keys in SECTIONS.items() for key in keys}
    grouped = {name: [] for name in [*SECTIONS, "other"]}
    for name, keys in SECTIONS.items():
        grouped[name] = [c for key in keys for c in controls if c.key == key]
    grouped["other"] = [c for c in controls if c.key not in known]
    return [(name, items) for name, items in grouped.items() if items]


class CameraTab(QWidget):
    def __init__(self, worker, config, set_camera, status_text):
        super().__init__()
        self.worker, self.config = worker, config
        self.set_camera, self.status_text = set_camera, status_text
        self.key = worker.camera.key
        self.rows = {}  # control key -> (control, widget, value label or None)
        self.controls = {}
        self.save_timer = QTimer(self, singleShot=True, interval=500, timeout=config.save)

        layout = QVBoxLayout(self)
        self.status = QLabel()
        layout.addWidget(self.status)
        self.blur = QCheckBox(tr("settings.blur"))
        self.blur.toggled.connect(lambda on: self.set_camera(self.key, "blur", on))
        layout.addWidget(self.blur)

        res_row = QHBoxLayout()
        res_row.addWidget(QLabel(tr("settings.resolution")))
        self.resolution = QComboBox()
        self.modes = [None] + v4l2.menu_modes(worker.modes)
        for _ in self.modes:
            self.resolution.addItem("")
        self.resolution.activated.connect(self.on_resolution)
        res_row.addWidget(self.resolution, 1)
        layout.addLayout(res_row)
        self.pending = QLabel(tr("settings.resolution_pending"))
        self.pending.setWordWrap(True)
        self.pending.setEnabled(False)
        layout.addWidget(self.pending)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        self.image = QVBoxLayout(inner)
        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)

        note = QLabel(tr("settings.image_note"))
        note.setWordWrap(True)
        note.setEnabled(False)
        layout.addWidget(note)
        reset = QPushButton(tr("settings.reset"))
        reset.clicked.connect(self.reset)
        buttons = QHBoxLayout()
        buttons.addWidget(reset)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.build_controls()
        self.refresh()

    # --- image controls ---

    def read_controls(self):
        try:
            return {c.key: c for c in v4l2.list_controls(self.worker.camera.path)}
        except OSError:
            return None

    def build_controls(self):
        controls = self.read_controls()
        if not controls:
            self.image.addWidget(QLabel(tr("settings.no_controls")))
            return
        self.controls = controls
        for section, items in sections(list(controls.values())):
            box = QGroupBox(tr(f"settings.section.{section}"))
            form = QFormLayout(box)
            for c in items:
                label = None
                if c.type == v4l2.CTRL_BOOLEAN or (c.minimum, c.maximum) == (0, 1):
                    widget = QCheckBox(control_name(c))
                    widget.toggled.connect(lambda on, k=c.key: self.on_change(k, int(on)))
                    form.addRow(widget)
                elif c.type == v4l2.CTRL_MENU:
                    widget = QComboBox()
                    for index in c.menu:
                        widget.addItem(menu_item_name(c, index), index)
                    widget.activated.connect(lambda i, k=c.key, w=widget: self.on_change(k, w.itemData(i)))
                    form.addRow(control_name(c), widget)
                else:
                    widget = QSlider(Qt.Orientation.Horizontal)
                    widget.setRange(c.minimum, c.maximum)
                    widget.setSingleStep(c.step)
                    widget.setPageStep(max(c.step, (c.maximum - c.minimum) // 10 // c.step * c.step))
                    label = QLabel()
                    label.setMinimumWidth(label.fontMetrics().horizontalAdvance("00000,0 ms"))
                    widget.valueChanged.connect(lambda v, k=c.key: self.on_slide(k, v))
                    row = QHBoxLayout()
                    row.addWidget(widget, 1)
                    row.addWidget(label)
                    form.addRow(control_name(c), row)
                self.rows[c.key] = (c, widget, label)
                if c.key == EXPOSURE:
                    self.exposure_hint = QLabel()
                    self.exposure_hint.setWordWrap(True)
                    form.addRow(self.exposure_hint)
            self.image.addWidget(box)
        self.image.addStretch(1)
        self.show_controls()

    def show_controls(self, skip=None):
        """Show the webcam's current values and which controls are active."""
        for key, (c, widget, label) in self.rows.items():
            current = self.controls.get(key, c)
            widget.setEnabled(not current.inactive)
            if key == skip or (isinstance(widget, QSlider) and widget.isSliderDown()):
                continue
            widget.blockSignals(True)
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(current.value))
            elif isinstance(widget, QComboBox):
                widget.setCurrentIndex(max(0, widget.findData(current.value)))
            else:
                widget.setValue(current.value)
                label.setText(value_text(c, current.value))
            widget.blockSignals(False)
        self.show_exposure_hint()

    def show_exposure_hint(self):
        if EXPOSURE not in self.rows:
            return
        c, widget, _ = self.rows[EXPOSURE]
        fps = self.worker.wanted_mode().fps if self.worker.modes else 30.0
        limit = round(10000 / fps)  # longest exposure that still fits in one frame
        too_long = widget.isEnabled() and widget.value() > limit
        self.exposure_hint.setText(tr("settings.exposure_slow" if too_long else "settings.exposure_limit",
                                      limit=value_text(c, limit), fps=f"{fps:g}"))
        self.exposure_hint.setEnabled(too_long)
        self.exposure_hint.setVisible(widget.isEnabled())

    def on_slide(self, key, value):
        c, widget, label = self.rows[key]
        value = round(value / c.step) * c.step if c.step > 1 else value
        label.setText(value_text(c, value))
        self.on_change(key, value)

    def on_change(self, key, value):
        c = self.rows[key][0]
        try:
            v4l2.set_control(self.worker.camera.path, c.id, value)
        except OSError:
            pass  # refused (e.g. inactive): the refresh below shows the real value
        self.config.camera(self.key)["controls"][key] = value
        self.save_timer.start()
        self.controls = self.read_controls() or self.controls  # auto modes change other controls
        self.show_controls(skip=key)

    def reset(self):
        saved = self.config.camera(self.key)["controls"]
        saved.clear()
        self.config.save()
        defaults = {key: c.default for key, (c, _w, _l) in self.rows.items()}
        try:
            v4l2.apply_controls(self.worker.camera.path, defaults)
        except OSError:
            pass
        self.controls = self.read_controls() or self.controls
        self.show_controls()

    # --- blur, resolution, status ---

    def on_resolution(self, index):
        mode = self.modes[index]
        self.set_camera(self.key, "mode", mode.to_config() if mode else None)

    def refresh(self):
        w = self.worker
        self.status.setText(self.status_text(w))
        self.blur.blockSignals(True)
        self.blur.setChecked(w.settings["blur"])
        self.blur.blockSignals(False)
        if w.modes:
            chosen = v4l2.find_mode(w.modes, w.settings.get("mode"))
            for i, mode in enumerate(self.modes):
                text = tr("settings.resolution_auto", mode=mode_text(v4l2.auto_mode(w.modes))) \
                    if mode is None else mode_text(mode)
                self.resolution.setItemText(i, text)
            keys = [m.key() if m else None for m in self.modes]
            chosen_key = chosen.key() if chosen else None
            self.resolution.setCurrentIndex(keys.index(chosen_key) if chosen_key in keys else 0)
        self.pending.setVisible(w.mode_pending)
        if self.isVisible() and self.rows:
            # Values changed elsewhere (another app, or the webcam's automatic modes).
            self.controls = self.read_controls() or self.controls
            self.show_controls()


class SettingsWindow(QWidget):
    def __init__(self, manager, config, set_camera, status_text):
        super().__init__()
        self.manager, self.config = manager, config
        self.set_camera, self.status_text = set_camera, status_text
        self.setWindowTitle(tr("settings.title"))
        self.resize(560, 720)
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self.empty = QLabel()
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.empty)
        close = QPushButton(tr("settings.close"))
        close.clicked.connect(self.close)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        self.layout_key = None
        self.refresh()

    def open(self):
        self.refresh()
        self.show()
        self.raise_()
        self.activateWindow()

    def refresh(self):
        workers = sorted(self.manager.workers.values(), key=lambda w: w.camera.name)
        key = tuple((w.camera.key, len(w.modes)) for w in workers)
        if key != self.layout_key:  # webcams plugged/unplugged, or their modes just listed
            self.layout_key = key
            current = self.tabs.currentWidget()
            current_key = current.key if current else None
            while self.tabs.count():
                tab = self.tabs.widget(0)
                self.tabs.removeTab(0)
                tab.deleteLater()
            for w in workers:
                tab = CameraTab(w, self.config, self.set_camera, self.status_text)
                self.tabs.addTab(tab, w.camera.name)
                if w.camera.key == current_key:
                    self.tabs.setCurrentWidget(tab)
        self.tabs.setVisible(bool(workers))
        self.empty.setVisible(not workers)
        self.empty.setText(self.manager.error or tr("menu.no_cameras"))
        for i in range(self.tabs.count()):
            self.tabs.widget(i).refresh()
