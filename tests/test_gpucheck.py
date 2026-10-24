import sys
import time

from webcam_blur import gpucheck, matting


def test_startup_check_runs_in_a_subprocess():
    """The tray-side check agrees with an in-process detection, without loading onnxruntime itself."""
    support = gpucheck.GpuSupport()
    assert support.state == "checking"
    done = []
    support.check(on_done=lambda: done.append(True))
    deadline = time.monotonic() + 120
    while not done:
        assert time.monotonic() < deadline, "GPU check never finished"
        time.sleep(0.05)
    expected = matting.detect_gpu()
    assert (support.state, support.provider) == (expected["state"], expected["provider"])


def test_tray_modules_stay_light():
    import subprocess
    code = ("import sys; import webcam_blur.app, webcam_blur.manager, webcam_blur.gpucheck; "
            "print(','.join(m for m in ('cv2', 'onnxruntime', 'numpy', 'onnx') if m in sys.modules))")
    from webcam_blur.proc import module_command
    _, env = module_command("app")
    env["QT_QPA_PLATFORM"] = "offscreen"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"the tray process must not import {out.stdout.strip()}"
