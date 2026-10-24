"""Resource use of the app and its GPU subprocesses, read from /proc and sysfs."""

import os
import time

KFD_PROCS = "/sys/class/kfd/kfd/proc"  # AMD ROCm: per-process VRAM
CLOCK_TICKS = os.sysconf("SC_CLK_TCK")
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


def our_pids():
    """This process and its direct children (the GPU subprocesses)."""
    me = os.getpid()
    pids = [me]
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                with open(f"/proc/{entry}/stat") as f:
                    # The command name (2nd field) may contain spaces: split after its ')'.
                    ppid = int(f.read().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            if ppid == me:
                pids.append(int(entry))
    return pids


def _cpu_ticks(pid):
    with open(f"/proc/{pid}/stat") as f:
        fields = f.read().rsplit(")", 1)[1].split()
    return int(fields[11]) + int(fields[12])  # utime + stime


def _rss(pid):
    with open(f"/proc/{pid}/statm") as f:
        return int(f.read().split()[1]) * PAGE_SIZE


def _vram(pid):
    """VRAM in bytes for ``pid``, or None if the driver does not report it."""
    folder = f"{KFD_PROCS}/{pid}"
    if not os.path.isdir(folder):
        return None
    total = 0
    for name in os.listdir(folder):
        if name.startswith("vram_"):
            try:
                with open(f"{folder}/{name}") as f:
                    total += int(f.read())
            except (OSError, ValueError):
                pass
    return total


class ResourceMeter:
    """CPU (% of the whole machine), RAM and VRAM of the app, between two samples."""

    def __init__(self):
        self.cpus = os.cpu_count() or 1
        self.ticks = {}  # pid -> cpu ticks at the previous sample
        self.when = None

    def sample(self):
        now = time.monotonic()
        ticks, ram, vram, vram_known = {}, 0, 0, os.path.isdir(KFD_PROCS)
        for pid in our_pids():
            try:
                ticks[pid] = _cpu_ticks(pid)
                ram += _rss(pid)
            except OSError:  # exited meanwhile
                continue
            used = _vram(pid)
            if used:
                vram += used
        cpu = 0.0
        if self.when is not None and now > self.when:
            # Processes that started since the last sample count from zero.
            delta = sum(t - self.ticks.get(pid, 0) for pid, t in ticks.items())
            cpu = max(0.0, delta / CLOCK_TICKS / (now - self.when) / self.cpus * 100)
        self.ticks, self.when = ticks, now
        return {"cpu": cpu, "ram": ram, "vram": vram if vram_known else None}
