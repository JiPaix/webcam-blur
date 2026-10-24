"""Preparing GPU models ahead of time (freezing; compiling is MIGraphX-only, see the stamp tests)."""

import time

import pytest

from webcam_blur import matting, prepare
from webcam_blur.models import GPU_MODELS
from webcam_blur.v4l2 import Mode

pytestmark = pytest.mark.usefixtures("models")

MBV3, R50 = GPU_MODELS["standard"], GPU_MODELS["high"]
MIGX = "MIGraphXExecutionProvider"


def test_jobs_cover_every_quality_current_one_first():
    jobs = prepare.jobs_for([Mode(1280, 720, 30.0)], "high", MIGX)
    assert jobs == [[R50, 1280, 720, MIGX], [MBV3, 640, 360, MIGX], [MBV3, 1280, 720, MIGX]]


def test_jobs_are_not_repeated():
    # At 640 px wide, "low" and "standard" are the same size; two webcams share a mode.
    jobs = prepare.jobs_for([Mode(640, 480, 30.0), Mode(640, 480, 30.0)], "low", MIGX)
    assert jobs == [[MBV3, 640, 480, MIGX], [R50, 640, 480, MIGX]]


def test_migraphx_models_need_a_matching_stamp(monkeypatch):
    frozen = matting.frozen_path(MBV3, 64, 36)
    frozen.parent.mkdir(parents=True, exist_ok=True)
    assert not prepare.is_prepared(MBV3, 64, 36, MIGX)
    frozen.write_bytes(b"")
    assert not prepare.is_prepared(MBV3, 64, 36, MIGX), "frozen but never compiled"
    cache = matting.migraphx_cache(frozen)
    cache.mkdir(parents=True)
    (cache / "stamp").write_text(prepare.stamp())
    assert prepare.is_prepared(MBV3, 64, 36, MIGX)
    monkeypatch.setattr(prepare, "stamp", lambda: "onnxruntime 99, ROCm 99\n")
    assert not prepare.is_prepared(MBV3, 64, 36, MIGX), "compiled by an older ROCm"


def wait_done(pre, timeout=120):
    deadline = time.monotonic() + timeout
    while pre.proc is not None:
        assert time.monotonic() < deadline, "preparation did not finish"
        time.sleep(0.2)
        pre.update([])
    return pre


def test_precompiler_prepares_in_the_background():
    provider = "CPUExecutionProvider"
    jobs = prepare.jobs_for([Mode(320, 180, 30.0)], "low", provider)
    assert not any(prepare.is_prepared(*job) for job in jobs)
    pre = prepare.Precompiler()
    pre.update(jobs)
    assert pre.proc is not None
    assert pre.proc.wait(120) == 0
    assert all(prepare.is_prepared(*job) for job in jobs)
    pre.update(jobs)
    assert pre.proc is None, "nothing left to do"
    pre.update(list(reversed(jobs)))
    assert pre.proc is None, "the same sizes in another order are done too"
    pre.update(prepare.jobs_for([Mode(320, 240, 30.0)], "low", provider))
    assert pre.proc is not None, "a new resolution needs preparing"
    wait_done(pre)


def test_failed_preparation_is_not_retried_in_a_loop(capsys):
    pre = prepare.Precompiler()
    jobs = [["no_such_model.onnx", 320, 180, "CPUExecutionProvider"]]
    pre.update(jobs)
    assert pre.proc.wait(60) == 1
    pre.update(jobs)
    assert pre.proc is None
    assert "model preparation failed" in capsys.readouterr().err
