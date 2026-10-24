"""The webcam settings window, on a simulated webcam (offscreen Qt, no device needed)."""

import pytest
from PyQt6.QtWidgets import QCheckBox, QComboBox, QSlider

from webcam_blur import settings, v4l2
from webcam_blur.cameras import Camera
from webcam_blur.config import Config
from webcam_blur.v4l2 import CTRL_BOOLEAN, CTRL_INTEGER, CTRL_MENU, Control, Mode

pytestmark = pytest.mark.usefixtures("qt_app")

MODES = [Mode(1280, 720, 60.0, "MJPG"), Mode(1280, 720, 30.0, "MJPG"), Mode(640, 480, 30.0, "MJPG")]
# key: (id, type, min, max, step, default, menu)
SPECS = {
    "brightness": (1, CTRL_INTEGER, 0, 255, 1, 128, {}),
    "backlight_compensation": (2, CTRL_INTEGER, 0, 1, 1, 0, {}),
    "auto_exposure": (3, CTRL_MENU, 0, 3, 1, 3, {1: "Manual Mode", 3: "Aperture Priority Mode"}),
    "exposure_time_absolute": (4, CTRL_INTEGER, 3, 2047, 1, 250, {}),
    "exposure_dynamic_framerate": (5, CTRL_BOOLEAN, 0, 1, 1, 0, {}),
    "led1_mode": (6, CTRL_MENU, 0, 1, 1, 0, {0: "Off", 1: "On"}),
}


class FakeWebcam:
    """Controls with UVC's rule: manual exposure is inactive (and refused) while auto."""

    def __init__(self):
        self.values = {key: spec[5] for key, spec in SPECS.items()}

    def controls(self, path):
        out = []
        for key, (cid, ctype, lo, hi, step, default, menu) in SPECS.items():
            inactive = key == "exposure_time_absolute" and self.values["auto_exposure"] == 3
            out.append(Control(cid, key, key.replace("_", " ").title(), ctype, lo, hi, step, default,
                               self.values[key], v4l2.CTRL_FLAG_INACTIVE if inactive else 0, menu))
        return out

    def set(self, path, cid, value):
        key = next(k for k, spec in SPECS.items() if spec[0] == cid)
        if key == "exposure_time_absolute" and self.values["auto_exposure"] == 3:
            raise OSError(13, "Permission denied")
        self.values[key] = value


class FakeWorker:
    def __init__(self, config, key="cam", name="Test Cam"):
        self.camera = Camera(key=key, name=name, path="/dev/video-test")
        self.settings = config.camera(key)
        self.modes = MODES
        self.mode, self.error, self.in_use, self.engine = None, None, False, None
        self.mode_pending = False

    def wanted_mode(self):
        return v4l2.find_mode(self.modes, self.settings.get("mode")) or v4l2.auto_mode(self.modes)


@pytest.fixture
def cam(monkeypatch, tmp_path):
    fake = FakeWebcam()
    monkeypatch.setattr(v4l2, "list_controls", fake.controls)
    monkeypatch.setattr(v4l2, "set_control", fake.set)
    config = Config(tmp_path / "config.json")
    calls = []

    def set_camera(key, setting, value):
        calls.append((key, setting, value))
        config.camera(key)[setting] = value

    worker = FakeWorker(config)
    tab = settings.CameraTab(worker, config, set_camera, lambda w: "status")
    return tab, fake, config, calls, worker


def widget(tab, key):
    return tab.rows[key][1]


def test_widget_per_control_type(cam):
    tab = cam[0]
    assert isinstance(widget(tab, "brightness"), QSlider)
    assert isinstance(widget(tab, "backlight_compensation"), QCheckBox), "0-1 integer: on/off"
    assert isinstance(widget(tab, "exposure_dynamic_framerate"), QCheckBox)
    combo = widget(tab, "auto_exposure")
    assert isinstance(combo, QComboBox)
    assert [combo.itemText(i) for i in range(combo.count())] == ["Manual", "Automatic"]  # translated
    assert combo.currentText() == "Automatic"


def test_unknown_controls_go_under_other():
    controls = FakeWebcam().controls(None)
    grouped = dict(settings.sections(controls))
    assert [c.key for c in grouped["other"]] == ["led1_mode"]
    assert [c.key for c in grouped["exposure"]] == [
        "auto_exposure", "exposure_time_absolute", "exposure_dynamic_framerate"]


def test_slider_sets_the_webcam_and_is_saved(cam):
    tab, fake, config, _, _ = cam
    widget(tab, "brightness").setValue(200)
    assert fake.values["brightness"] == 200
    assert config.camera("cam")["controls"] == {"brightness": 200}
    assert tab.rows["brightness"][2].text() == "200"
    assert tab.save_timer.isActive(), "saved shortly after, not on every slider step"


def test_manual_exposure_follows_auto_mode(cam):
    tab, fake, config, _, _ = cam
    exposure = widget(tab, "exposure_time_absolute")
    assert not exposure.isEnabled(), "inactive while exposure is automatic"
    combo = widget(tab, "auto_exposure")
    combo.setCurrentIndex(combo.findData(1))
    combo.activated.emit(combo.currentIndex())
    assert fake.values["auto_exposure"] == 1
    assert exposure.isEnabled()
    exposure.setValue(400)
    assert fake.values["exposure_time_absolute"] == 400
    assert tab.rows["exposure_time_absolute"][2].text() == "40.0 ms"
    assert config.camera("cam")["controls"] == {"auto_exposure": 1, "exposure_time_absolute": 400}


def test_exposure_hint_shows_the_frame_rate_limit(cam):
    tab, fake, config, _, worker = cam
    fake.values["auto_exposure"] = 1
    tab.controls = tab.read_controls()
    tab.show_controls()
    worker.settings["mode"] = MODES[0].to_config()  # 60 fps: one frame lasts 16.7 ms
    widget(tab, "exposure_time_absolute").setValue(150)
    assert tab.exposure_hint.text() == "Up to 16.7 ms keeps 60 fps."
    widget(tab, "exposure_time_absolute").setValue(400)
    assert tab.exposure_hint.text() == "Longer than 16.7 ms: the webcam drops below 60 fps."


def test_reset_restores_defaults_and_forgets_saved_values(cam):
    tab, fake, config, _, _ = cam
    widget(tab, "brightness").setValue(10)
    widget(tab, "exposure_dynamic_framerate").setChecked(True)
    tab.reset()
    assert fake.values == {key: spec[5] for key, spec in SPECS.items()}
    assert config.camera("cam")["controls"] == {}
    assert widget(tab, "brightness").value() == 128


def test_blur_and_resolution(cam):
    tab, _, _, calls, _ = cam
    tab.blur.setChecked(False)
    assert calls[-1] == ("cam", "blur", False)
    assert tab.resolution.itemText(0) == "Automatic (1280 × 720 · 30 fps)"
    tab.resolution.activated.emit(1)  # first listed mode
    assert calls[-1] == ("cam", "mode", {"width": 1280, "height": 720, "fps": 60.0})


def test_window_has_a_tab_per_webcam(monkeypatch, tmp_path):
    monkeypatch.setattr(v4l2, "list_controls", FakeWebcam().controls)
    config = Config(tmp_path / "config.json")

    class Manager:
        error = None
        workers = {}

    manager = Manager()
    window = settings.SettingsWindow(manager, config, lambda *a: None, lambda w: "status")
    assert window.tabs.count() == 0 and not window.empty.isHidden()
    manager.workers = {"b": FakeWorker(config, "b", "B cam"), "a": FakeWorker(config, "a", "A cam")}
    window.refresh()
    assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == ["A cam", "B cam"]
    assert window.empty.isHidden()
