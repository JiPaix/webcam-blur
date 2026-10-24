import multiprocessing as mp
import time

from webcam_blur import i18n
from webcam_blur.app import resources_text
from webcam_blur.resources import ResourceMeter, our_pids


def _busy(seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        pass


def test_counts_the_app_and_its_subprocesses():
    meter = ResourceMeter()
    idle = meter.sample()
    assert idle["cpu"] == 0.0 and idle["ram"] > 10e6
    child = mp.get_context("spawn").Process(target=_busy, args=(3,))
    child.start()
    try:
        time.sleep(1.5)
        assert child.pid in our_pids()
        busy = meter.sample()
        # One busy core out of N, measured over ~1.5 s.
        assert busy["cpu"] > 50 / meter.cpus, busy
        assert busy["ram"] > idle["ram"], "the subprocess's memory must be included"
    finally:
        child.join()


def test_vram_is_none_or_bytes():
    vram = ResourceMeter().sample()["vram"]
    assert vram is None or vram >= 0


def test_text_is_localised():
    usage = {"cpu": 4.25, "ram": 412e6, "vram": 1.43e9}
    i18n.setup("en")
    assert resources_text(usage) == "CPU 4.2 % · RAM 412 MB · VRAM 1.4 GB"
    assert resources_text({**usage, "vram": 910e6}) == "CPU 4.2 % · RAM 412 MB · VRAM 910 MB"
    i18n.setup("fr")
    assert resources_text(usage) == "CPU 4,2 % · RAM 412 Mo · VRAM 1,4 Go"
    assert resources_text({**usage, "vram": None, "cpu": 23.0}) == "CPU 23 % · RAM 412 Mo"


def test_vram_counts_each_gpu_client_once(monkeypatch, tmp_path):
    """amdgpu reports the same memory as drm-total-vram and drm-memory-vram."""
    import os

    from webcam_blur import resources
    fdinfo = {"6": "drm-client-id:\t7\ndrm-total-vram:\t226880 KiB\ndrm-resident-vram:\t226880 KiB\n"
                   "drm-memory-vram:\t226880 KiB\n",
              "7": "drm-client-id:\t7\ndrm-total-vram:\t226880 KiB\n",  # same client, another fd
              "8": "pos:\t0\n"}  # not a GPU
    real_listdir, real_open = os.listdir, open
    monkeypatch.setattr(resources.os, "listdir", lambda p: list(fdinfo) if p.endswith("/fdinfo") else real_listdir(p))

    class Fake:
        def __init__(self, text): self.text = text
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return self.text

    monkeypatch.setattr("builtins.open", lambda p, *a, **k: Fake(fdinfo[p.rsplit("/", 1)[1]])
                        if "/fdinfo/" in str(p) else real_open(p, *a, **k))
    assert resources._vram(1234, set()) == 226880 * 1024
