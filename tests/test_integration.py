"""End-to-end tests against the real v4l2loopback kernel module.

A fake webcam (a loopback device fed with the test portrait at 30 fps) is handed to
the real Manager, which must create the virtual camera, stay idle until an app reads
it, blur the background while it does, react to the webcam being unplugged, and
clean up. "Apps" are separate processes (tests/consumer.py), as in real use.

Needs the v4l2loopback module (>= 0.13, loaded with devices=0 or any count) and
read/write access to /dev/v4l2loopback; in CI these tests run as root.
"""

import json
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import FRAME, PERSON, TEXTURED, detail, mean_abs_diff

from webcam_blur import cameras, matting, v4l2
from webcam_blur.cameras import Camera
from webcam_blur.config import Config
from webcam_blur.worker import IDLE_GRACE

pytestmark = [pytest.mark.v4l2, pytest.mark.usefixtures("models")]

CONSUMER = Path(__file__).parent / "consumer.py"
MIN_FPS = float(os.environ.get("WEBCAM_BLUR_TEST_MIN_FPS", "10"))
LABEL = "Test Cam (blur)"


def wait_for(condition, timeout, interval=0.1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return condition()


def our_handles(path):
    """How many file descriptors this process has open on ``path``."""
    target = os.path.realpath(path)
    count = 0
    for fd in os.listdir("/proc/self/fd"):
        try:
            count += os.readlink(f"/proc/self/fd/{fd}") == target
        except OSError:
            pass
    return count


def webcam_opened_by_worker(webcam):
    # The fake webcam's own feeder keeps one handle open; the worker's capture adds another.
    return our_handles(webcam.path) > 1


def start_consumer(device, seconds, out):
    return subprocess.Popen([sys.executable, str(CONSUMER), device, str(seconds), str(out)],
                            stdout=subprocess.PIPE, text=True)


def finish_consumer(proc, out):
    stdout, _ = proc.communicate(timeout=120)
    stats = json.loads(stdout)
    frame = np.load(out) if Path(out).exists() else None
    return stats, frame


def consume(device, seconds, tmp_path):
    out = tmp_path / f"frame-{time.monotonic_ns()}.npy"
    return finish_consumer(start_consumer(device, seconds, out), out)


def labels(control):
    return list(control.devices().values())


class FakeWebcam:
    """A loopback device fed with the test frame, standing in for a real webcam."""

    def __init__(self, control, frame):
        self.control = control
        self.nr = control.add("webcam-blur test input")
        self.path = f"/dev/video{self.nr}"
        self.out = v4l2.LoopbackOutput(self.path, frame.shape[1], frame.shape[0])
        self.frame = frame
        self.out.write(frame)  # a loopback device advertises its format only after a first frame
        self.stop_evt = threading.Event()
        self.thread = threading.Thread(target=self.feed, daemon=True)
        self.thread.start()

    def feed(self):
        while not self.stop_evt.is_set():
            self.out.write(self.frame)
            time.sleep(1 / 30)

    def close(self):
        self.stop_evt.set()
        self.thread.join()
        self.out.close()
        wait_for(lambda: self.control.remove(self.nr), 10, 0.5)


@pytest.fixture(scope="module")
def qt_app():
    from PyQt6.QtCore import QCoreApplication
    return QCoreApplication.instance() or QCoreApplication([])


@pytest.fixture(scope="module")
def control():
    try:
        ctl = v4l2.LoopbackControl()
    except OSError as e:
        pytest.skip(f"/dev/v4l2loopback not usable: {e}")
    yield ctl
    ctl.close()


@pytest.fixture(scope="module")
def webcam(control):
    fake = FakeWebcam(control, FRAME)
    yield fake
    fake.close()


@pytest.fixture(scope="module")
def pipeline(qt_app, control, webcam, tmp_path_factory):
    from webcam_blur.manager import Manager
    config = Config(tmp_path_factory.mktemp("config") / "config.json")
    config["use_gpu"] = False
    cam = Camera(key="test-cam", name="Test Cam", path=webcam.path)
    state = {"present": True}
    manager = Manager(config, scan=lambda: {cam.key: cam} if state["present"] else {})
    manager.sync()
    yield SimpleNamespace(manager=manager, config=config, cam=cam, state=state, control=control)
    manager.shutdown()


def worker_of(pipeline):
    return pipeline.manager.workers[pipeline.cam.key]


# --- Kernel interfaces ----------------------------------------------------------

def test_control_device_add_query_remove(control):
    nr = control.add("webcam-blur probe")
    try:
        assert os.path.exists(f"/dev/video{nr}")
        assert control.query(nr) == "webcam-blur probe"
        assert control.devices()[nr] == "webcam-blur probe"
    finally:
        assert control.remove(nr)
    assert nr not in control.devices()
    assert control.query(nr) is None


def test_fake_webcam_reports_its_mode(webcam):
    modes = v4l2.capture_modes(webcam.path)
    assert v4l2.pick_mode(modes) == ("YUYV", 1280, 720)


def test_hotplug_monitor_sees_new_devices(control):
    if not os.path.exists("/run/udev/control"):
        pytest.skip("no udev daemon")
    events = []
    monitor = cameras.HotplugMonitor(lambda: events.append(time.monotonic()))
    monitor.start()
    try:
        time.sleep(0.5)
        nr = control.add("webcam-blur hotplug probe")
        assert wait_for(lambda: events, 5), "no udev event for a new video device"
        control.remove(nr)
    finally:
        monitor.stop()


def test_real_scan_ignores_virtual_devices(webcam):
    assert all(cam.path != webcam.path for cam in cameras.scan().values())


# --- Full pipeline ----------------------------------------------------------------

def test_virtual_camera_created(pipeline):
    worker = worker_of(pipeline)
    assert pipeline.control.query(worker.device_nr) == LABEL
    assert os.path.exists(worker.output_path)


def test_webcam_stays_closed_while_nobody_reads(pipeline, webcam):
    time.sleep(2.5)
    worker = worker_of(pipeline)
    assert worker.error is None
    assert not worker.in_use
    assert not webcam_opened_by_worker(webcam), "the real webcam must only be opened while an app reads"


def test_app_gets_blurred_frames(pipeline, webcam, tmp_path):
    worker = worker_of(pipeline)
    out_path = tmp_path / "blur.npy"
    proc = start_consumer(worker.output_path, 8, out_path)
    assert wait_for(lambda: worker.in_use, 5), "an app reading the virtual camera must start the stream"
    assert webcam_opened_by_worker(webcam)
    stats, out = finish_consumer(proc, out_path)
    print(f"\nconsumer stats: {stats}")
    assert stats["frames"] > 0, "no frames from the virtual camera"
    assert stats["fps_second_half"] >= MIN_FPS, f"too slow: {stats['fps_second_half']:.1f} fps"
    assert out.shape == FRAME.shape
    for name in PERSON:
        assert mean_abs_diff(out, FRAME, name) < 10, f"{name} should be untouched"
    for name in TEXTURED:
        assert detail(out, name) < 0.3 * detail(FRAME, name), f"{name} should be blurred"
    # Once the app is gone, the real webcam is released.
    assert wait_for(lambda: not worker.in_use, IDLE_GRACE + 5)
    assert wait_for(lambda: not webcam_opened_by_worker(webcam), 3)


def test_blur_off_passes_the_image_through(pipeline, tmp_path):
    settings = pipeline.config.camera(pipeline.cam.key)
    settings["blur"] = False
    try:
        _, out = consume(worker_of(pipeline).output_path, 4, tmp_path)
    finally:
        settings["blur"] = True
    for name in TEXTURED:
        assert detail(out, name) > 0.7 * detail(FRAME, name), f"{name} should not be blurred"
    for name in PERSON:
        assert mean_abs_diff(out, FRAME, name) < 10
    assert wait_for(lambda: not worker_of(pipeline).in_use, IDLE_GRACE + 5)


def test_gpu_path_in_live_pipeline(pipeline, tmp_path, monkeypatch):
    """The GPU subprocess (here on onnxruntime's CPU provider) takes over, then goes away."""
    monkeypatch.setattr(matting.gpu_support, "state", "ok")
    monkeypatch.setattr(matting.gpu_support, "provider", "CPUExecutionProvider")
    pipeline.config.data["use_gpu"] = True
    worker = worker_of(pipeline)
    out_path = tmp_path / "gpu.npy"
    try:
        proc = start_consumer(worker.output_path, 45, out_path)
        assert wait_for(lambda: worker.engine == "gpu", 40, 0.25), f"engine stayed {worker.engine!r}"
        stats, out = finish_consumer(proc, out_path)
    finally:
        pipeline.config.data["use_gpu"] = False
    print(f"\nconsumer stats (GPU path on CPU provider): {stats}")
    for name in TEXTURED:
        assert detail(out, name) < 0.3 * detail(FRAME, name), f"{name} should be blurred"
    assert wait_for(lambda: not worker.in_use, IDLE_GRACE + 5)
    assert wait_for(lambda: not mp.active_children(), 5), "GPU subprocess still running after the stream"
    assert not [f for f in os.listdir("/dev/shm") if f.startswith("psm_")], "shared memory leaked"


def test_unplug_and_replug(pipeline):
    pipeline.state["present"] = False
    pipeline.manager.sync()
    assert pipeline.cam.key not in pipeline.manager.workers
    assert LABEL not in labels(pipeline.control), "virtual camera should go with its webcam"
    pipeline.state["present"] = True
    pipeline.manager.sync()
    assert pipeline.cam.key in pipeline.manager.workers
    assert LABEL in labels(pipeline.control)


def test_virtual_camera_in_use_is_removed_once_released(pipeline, tmp_path):
    worker = worker_of(pipeline)
    out = tmp_path / "busy.npy"
    proc = start_consumer(worker.output_path, 6, out)
    assert wait_for(lambda: worker.in_use, 10)
    pipeline.state["present"] = False
    pipeline.manager.sync()
    assert pipeline.cam.key in pipeline.manager.orphans, "removal must wait while an app holds the device"
    assert LABEL in labels(pipeline.control)
    finish_consumer(proc, out)
    pipeline.manager.sync()
    assert pipeline.cam.key not in pipeline.manager.orphans
    assert LABEL not in labels(pipeline.control)
    pipeline.state["present"] = True
    pipeline.manager.sync()
    assert LABEL in labels(pipeline.control)


def test_leftover_virtual_camera_is_reused(qt_app, control, webcam, tmp_path):
    from webcam_blur.manager import Manager
    cam = Camera(key="second-cam", name="Second Cam", path=webcam.path)
    leftover = control.add("Second Cam (blur)")
    manager = Manager(Config(tmp_path / "config.json"), scan=lambda: {cam.key: cam})
    try:
        manager.start()
        assert manager.workers[cam.key].device_nr == leftover
        assert list(control.devices().values()).count("Second Cam (blur)") == 1
    finally:
        manager.shutdown()
    assert "Second Cam (blur)" not in labels(control)


def test_shutdown_removes_virtual_cameras(pipeline):
    pipeline.manager.shutdown()
    assert not pipeline.manager.workers
    assert LABEL not in labels(pipeline.control)
