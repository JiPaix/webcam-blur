"""Resource use of the app and its subprocesses, read from /proc and sysfs."""

import os
import time

# GPU memory per process, from the DRM driver's fdinfo (amdgpu, i915/xe, nouveau…), in KiB.
VRAM_KEYS = ("drm-memory-vram:", "drm-total-vram", "drm-total-local")
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



def _vram(pid, seen):
    """VRAM in bytes used by ``pid``'s GPU clients, or None if it has none the driver reports.
    ``seen``: DRM client ids already counted (a client can be open in several fds)."""
    total = None
    try:
        fds = os.listdir(f"/proc/{pid}/fdinfo")
    except OSError:
        return None
    for fd in fds:
        try:
            with open(f"/proc/{pid}/fdinfo/{fd}") as f:
                lines = f.read().splitlines()
        except OSError:
            continue  # closed meanwhile
        client = next((line.split(":", 1)[1].strip() for line in lines if line.startswith("drm-client-id:")), None)
        if client is None or client in seen:
            continue
        seen.add(client)
        for line in lines:
            if line.startswith(VRAM_KEYS):
                value = line.split(":", 1)[1].split()
                total = (total or 0) + int(value[0]) * (1024 if value[1:] == ["KiB"] else 1)
    return total


class ResourceMeter:
    """CPU (% of the whole machine), RAM and VRAM of the app, between two samples."""

    def __init__(self):
        self.cpus = os.cpu_count() or 1
        self.ticks = {}  # pid -> cpu ticks at the previous sample
        self.when = None

    def sample(self):
        now = time.monotonic()
        ticks, ram, vram, seen = {}, 0, None, set()
        for pid in our_pids():
            try:
                ticks[pid] = _cpu_ticks(pid)
                ram += _pss(pid)
            except OSError:  # exited meanwhile
                continue
            used = _vram(pid, seen)
            if used is not None:
                vram = (vram or 0) + used
        cpu = 0.0
        if self.when is not None and now > self.when:
            # Processes that started since the last sample count from zero.
            delta = sum(t - self.ticks.get(pid, 0) for pid, t in ticks.items())
            cpu = max(0.0, delta / CLOCK_TICKS / (now - self.when) / self.cpus * 100)
        self.ticks, self.when = ticks, now
        return {"cpu": cpu, "ram": ram, "vram": vram}  # vram: None while nothing uses the GPU
