"""Resource use of the app and its subprocesses, read from /proc and sysfs."""

import os
import time

KFD_PROCS = "/sys/class/kfd/kfd/proc"  # AMD ROCm: per-process VRAM
CLOCK_TICKS = os.sysconf("SC_CLK_TCK")


def our_pids():
    """This process and all its descendants (stream subprocesses and their GPU subprocesses)."""
    children = {}
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                with open(f"/proc/{entry}/stat") as f:
                    # The command name (2nd field) may contain spaces: split after its ')'.
                    ppid = int(f.read().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            children.setdefault(ppid, []).append(int(entry))
    pids, todo = [], [os.getpid()]
    while todo:
        pid = todo.pop()
        pids.append(pid)
        todo.extend(children.get(pid, ()))
    return pids


def _cpu_ticks(pid):
    with open(f"/proc/{pid}/stat") as f:
        fields = f.read().rsplit(")", 1)[1].split()
    return int(fields[11]) + int(fields[12])  # utime + stime


def _pss(pid):
    """Proportional memory: private memory plus a share of the libraries used by several
    processes (Qt, for one, is shared with the desktop)."""
    with open(f"/proc/{pid}/smaps_rollup") as f:
        for line in f:
            if line.startswith("Pss:"):
                return int(line.split()[1]) * 1024
    return 0



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
                ram += _pss(pid)
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
