"""End-to-end tests against the real v4l2loopback kernel module.

A fake webcam (a loopback device fed with the test portrait at 30 fps) is handed to
the real Manager, which must create the virtual camera, stay idle until an app reads
it, blur the background while it does, react to the webcam being unplugged, and
clean up. "Apps" are separate processes (tests/consumer.py), as in real use.

Needs the v4l2loopback module (>= 0.13, loaded with devices=0 or any count) and
read/write access to /dev/v4l2loopback; in CI these tests run as root.
"""

import json
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

from webcam_blur import cameras, gpucheck, v4l2
from webcam_blur.cameras import Camera
from webcam_blur.config import Config
from webcam_blur.resources import our_pids
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


def webcam_opened_by_worker(webcam):
    # The fake webcam's feeder runs in this test process, which readers_of() leaves out:
    # any reader is the stream subprocess's capture.
    return v4l2.readers_of(webcam.path) > 0


def cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


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


def advertised_fps(path):
    """Frame rate a newly opening app would read (VIDIOC_G_PARM on the capture side)."""
    import fcntl
    import struct
    fd = os.open(path, os.O_RDWR)
    try:
        parm = bytearray(struct.pack("<I", v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE) + b"\0" * 200)
        fcntl.ioctl(fd, 0xC0CC5615, parm, True)  # VIDIOC_G_PARM
        num, den = struct.unpack_from("<II", parm, 12)
        return den / num if num else None
    except OSError:  # EINVAL while the device is being reconfigured
        return None
    finally:
        os.close(fd)


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


TEST_LABELS = (LABEL, "Second Cam (blur)", "webcam-blur test input", "webcam-blur probe",
               "webcam-blur hotplug probe")


@pytest.fixture(scope="module")
def control():
    try:
        ctl = v4l2.LoopbackControl()
    except OSError as e:
        pytest.skip(f"/dev/v4l2loopback not usable: {e}")
    # Devices left behind by an interrupted earlier run would make label checks lie.
    for nr, label in ctl.devices().items():
        if label in TEST_LABELS:
            assert wait_for(lambda nr=nr: ctl.remove(nr), 30, 0.5), f"stale test device {label!r} is still in use"
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
    modes = v4l2.list_modes(webcam.path)
    assert v4l2.auto_mode(modes).key()[:2] == (1280, 720)
    assert all(m.fourcc == "YUYV" for m in modes)


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
    assert wait_for(lambda: webcam_opened_by_worker(webcam), 10), "the stream should open the webcam"
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
    assert worker.proc is None, "the stream subprocess (and its models) must be gone when idle"


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


def test_resolution_choice(pipeline, tmp_path):
    """A mode picked in the menu reconfigures the virtual camera, but never under a reading app."""
    worker = worker_of(pipeline)
    settings = pipeline.config.camera(pipeline.cam.key)
    auto = worker.mode
    small = v4l2.Mode(640, 360, 30.0, "YUYV")
    # The fake webcam only produces 1280x720; frames are scaled to the chosen size.
    worker.modes = v4l2.sort_modes(worker.modes + [small])
    try:
        settings["mode"] = small.to_config()
        assert wait_for(lambda: worker.mode == small, 5), "idle virtual camera should switch right away"
        _, out = consume(worker.output_path, 3, tmp_path)
        assert out.shape == (360, 640, 3)
        # The consumer (OpenCV) requested 30 fps on the shared rate; once idle, the chosen rate is back.
        settings["mode"] = v4l2.Mode(640, 360, 15.0, "YUYV").to_config()
        worker.modes = v4l2.sort_modes(worker.modes + [v4l2.Mode(640, 360, 15.0, "YUYV")])
        assert wait_for(lambda: advertised_fps(worker.output_path) == 15.0, 6)
        _, out = consume(worker.output_path, 2, tmp_path)
        assert wait_for(lambda: advertised_fps(worker.output_path) == 15.0, IDLE_GRACE + 5)
        worker.modes = [m for m in worker.modes if m.fps != 15.0]
        settings["mode"] = small.to_config()
        assert wait_for(lambda: worker.mode == small, 5)

        # While an app reads, the change waits and the menu says so.
        out_path = tmp_path / "pending.npy"
        proc = start_consumer(worker.output_path, 5, out_path)
        assert wait_for(lambda: worker.in_use, 5)
        settings["mode"] = None
        time.sleep(1.5)
        assert worker.mode == small and worker.mode_pending
        _, out = finish_consumer(proc, out_path)
        assert out.shape == (360, 640, 3), "the reading app must keep its format"
        assert wait_for(lambda: worker.mode == auto, IDLE_GRACE + 5), "switch once the app is gone"
        assert not worker.mode_pending
    finally:
        settings["mode"] = None
        worker.modes = [m for m in worker.modes if m != small]
    assert wait_for(lambda: worker.mode == auto, IDLE_GRACE + 5)


def test_gpu_path_in_live_pipeline(pipeline, tmp_path, monkeypatch):
    """The GPU subprocess (here on onnxruntime's CPU provider) takes over, then goes away."""
    monkeypatch.setattr(gpucheck.gpu_support, "state", "ok")
    monkeypatch.setattr(gpucheck.gpu_support, "provider", "CPUExecutionProvider")
    pipeline.config.data["use_gpu"] = True
    worker = worker_of(pipeline)
    out_path = tmp_path / "gpu.npy"
    try:
        proc = start_consumer(worker.output_path, 60, out_path)
        assert wait_for(lambda: worker.engine == "gpu", 40, 0.25), f"engine stayed {worker.engine!r}"
        # Switching quality mid-stream restarts the GPU subprocess with the other model.
        pipeline.config.data["quality"] = "high"
        assert wait_for(lambda: worker.engine != "gpu", 5, 0.1), "quality change should restart the GPU model"
        assert wait_for(lambda: worker.engine == "gpu", 45, 0.25), f"engine stayed {worker.engine!r}"
        stats, out = finish_consumer(proc, out_path)
    finally:
        pipeline.config.data["use_gpu"] = False
        pipeline.config.data["quality"] = "standard"
    print(f"\nconsumer stats (GPU path on CPU provider): {stats}")
    for name in TEXTURED:
        assert detail(out, name) < 0.3 * detail(FRAME, name), f"{name} should be blurred"
    assert wait_for(lambda: not worker.in_use, IDLE_GRACE + 5)
    assert worker.proc is None, "stream subprocess still running after the stream"
    assert not [p for p in our_pids() if "webcam_blur.stream" in cmdline(p)]
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
