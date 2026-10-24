"""Stand-in for "an app using the virtual camera", run as a separate process.

    python consumer.py DEVICE SECONDS OUT.npy

Reads frames for SECONDS, saves the last frame to OUT.npy and prints JSON stats.
"""

import json
import sys
import time

import cv2
import numpy as np

device, seconds, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
start = time.monotonic()
stamps, last = [], None
while time.monotonic() - start < seconds:
    ok, frame = cap.read()
    if ok:
        stamps.append(time.monotonic() - start)
        last = frame
    else:
        time.sleep(0.01)
cap.release()
if last is not None:
    np.save(out, last)
tail = [t for t in stamps if t >= seconds / 2]
print(json.dumps({
    "opened": cap is not None,
    "frames": len(stamps),
    "first_frame_after": stamps[0] if stamps else None,
    "fps_second_half": len(tail) / (seconds / 2),
}))
