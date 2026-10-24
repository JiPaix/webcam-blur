"""Webcam discovery and hotplug, via udev."""

import os
from dataclasses import dataclass

import pyudev


@dataclass(frozen=True)
class Camera:
    key: str      # stable identifier (by-id link name), used in the config
    name: str     # human-readable product name
    path: str     # stable device path to open (by-id/by-path link, or /dev/videoN)


def _camera_from(device):
    """Camera for a udev video4linux device, or None if it is not a real webcam capture node."""
    if device.device_node is None or "/devices/virtual/" in device.sys_path:
        return None  # v4l2loopback and other virtual devices (including ours)
    if ":capture:" not in device.properties.get("ID_V4L_CAPABILITIES", ""):
        return None  # metadata nodes, outputs, radio…
    if device.attributes.get("index") not in (b"0", None):
        return None  # secondary nodes of the same webcam
    links = device.properties.get("DEVLINKS", "").split()
    stable = next((ln for ln in links if ln.startswith("/dev/v4l/by-id/")), None) \
        or next((ln for ln in links if ln.startswith("/dev/v4l/by-path/")), None)
    name = device.properties.get("ID_V4L_PRODUCT") \
        or (device.attributes.get("name") or b"").decode(errors="replace") or device.sys_name
    key = os.path.basename(stable) if stable else device.sys_name
    return Camera(key=key, name=name.strip(), path=stable or device.device_node)


def scan(context=None):
    """{key: Camera} for every webcam currently plugged in."""
    context = context or pyudev.Context()
    found = {}
    for device in context.list_devices(subsystem="video4linux"):
        cam = _camera_from(device)
        if cam:
            found[cam.key] = cam
    return found


class HotplugMonitor:
    """Calls ``callback()`` (from a background thread) whenever a video device appears or goes."""

    def __init__(self, callback):
        context = pyudev.Context()
        monitor = pyudev.Monitor.from_netlink(context)
        monitor.filter_by("video4linux")
        self.observer = pyudev.MonitorObserver(monitor, callback=lambda _device: callback())
        self.observer.daemon = True

    def start(self):
        self.observer.start()

    def stop(self):
        self.observer.send_stop()
