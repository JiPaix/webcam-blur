# Webcam Blur

[![tests](https://github.com/JiPaix/webcam-blur/actions/workflows/tests.yml/badge.svg)](https://github.com/JiPaix/webcam-blur/actions/workflows/tests.yml)

Background blur for any webcam on Linux, usable in **every** app (Discord, Zoom,
Teams, Meet, OBS, browsers…).

For each webcam you plug in, Webcam Blur creates a virtual camera named
`<your webcam> (blur)`. Pick it in your app instead of the real webcam. In the tray
icon's menu, each webcam has a **Blur** switch and a **Resolution** list, which shows
every resolution and frame rate it supports, e.g. 1080p · 30 fps or 720p · 60 fps.
Blur strength is set for all cameras at once.

- **Automatic:** webcams are detected when plugged in. Their virtual cameras appear and
  disappear with them.
- **Only works when needed:** the real webcam is opened only while an app reads
  the virtual camera. The rest of the time nothing runs, so the webcam LED stays off and
  the GPU is untouched.
- **GPU accelerated:** uses Robust Video Matting (ResNet50) at full resolution on
  AMD (MIGraphX) or NVIDIA (CUDA) GPUs. Falls back to a lighter model on the CPU.
- **Translatable:** English and French included, other languages welcome (see
  [CONTRIBUTING.md](CONTRIBUTING.md)).

## How it works

```
real webcam ──▶ webcam-blur ──▶ v4l2loopback virtual camera ──▶ any app
                 (person cut out with RVM, background blurred)
```

- Virtual cameras are [v4l2loopback](https://github.com/v4l2loopback/v4l2loopback)
  devices. Webcam Blur creates them at runtime through `/dev/v4l2loopback`. A udev
  rule gives the logged-in user access to it.
- People are cut out with [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting)
  (ONNX Runtime).
- GPU execution providers such as MIGraphX need fixed input sizes. The ResNet50
  model is therefore converted once per camera resolution and cached in
  `~/.cache/webcam-blur/`.
- With MIGraphX, the first use of a resolution compiles the model (about 1–2 minutes,
  and again after ROCm updates). The CPU model is used until compilation finishes.
- GPU inference runs in a subprocess that exists only while a camera is being read.

## Installation

### Arch Linux and derivatives

A PKGBUILD is in [`packaging/arch`](packaging/arch):

```sh
git clone https://github.com/JiPaix/webcam-blur
cd webcam-blur/packaging/arch
makepkg -si
```

pacman asks which onnxruntime build to install: pick `python-onnxruntime-rocm` for an
AMD GPU, `python-onnxruntime-cuda` for NVIDIA, or `python-onnxruntime-cpu`. You can
switch later, e.g. `sudo pacman -S python-onnxruntime-rocm`.

At startup the app runs a tiny test model on the GPU. *Use GPU* only appears in the
menu if that test works. Otherwise the CPU model is used. The reason is logged
(`journalctl --user -t webcam-blur` or the terminal).

Then start **Webcam Blur** from your app menu. Tick *Start at login* in its tray menu
to keep it running.

The `v4l2loopback` kernel module (**version 0.13 or newer**) ships with some kernels
(e.g. CachyOS). For other kernels, install `v4l2loopback-dkms`. Ubuntu 24.04 only
packages 0.12, which cannot create virtual cameras at runtime: build a newer
v4l2loopback from source there.

### Other distributions / from source

Requirements: Python ≥ 3.11, PyQt6, OpenCV, NumPy, pyudev, onnx, onnxruntime (a GPU
build such as `onnxruntime-gpu` for NVIDIA, or ROCm/MIGraphX for AMD), and the
v4l2loopback module (≥ 0.13).

1. Install the system files (as root):

   ```sh
   install -Dm644 data/60-webcam-blur.rules /etc/udev/rules.d/60-webcam-blur.rules
   install -Dm644 data/modprobe.conf /etc/modprobe.d/webcam-blur.conf
   install -Dm644 data/modules-load.conf /etc/modules-load.d/webcam-blur.conf
   udevadm control --reload && modprobe v4l2loopback
   ```

2. Run the app (it downloads the models to `~/.local/share/webcam-blur/models` on
   first start):

   ```sh
   python -m webcam_blur
   ```

## Configuration

Everything is in the tray menu. Settings are stored in
`~/.config/webcam-blur/config.json`:

| Key | Meaning |
| --- | --- |
| `strength` | `light`, `medium` or `strong` |
| `use_gpu` | `true` / `false` |
| `language` | e.g. `"fr"`; `null` follows the system locale |
| `cameras.<id>.blur` | blur on/off for this camera |
| `cameras.<id>.virtual` | `false` to never create a virtual camera for this webcam (greyscale-only devices such as IR face-login cameras are skipped automatically) |
| `cameras.<id>.mode` | resolution chosen in the menu (`{"width", "height", "fps"}`), `null` = automatic |

**Automatic resolution:** 1280×720 at 30 fps if the webcam offers it; otherwise the next
size up that reaches 30 fps (e.g. 1080p30). Calling apps rarely send more, and every
extra pixel costs processing. A resolution chosen in the menu takes effect once no
app is using the camera.

**Picture settings** (brightness, exposure, white balance, focus, zoom…) are stored on
the webcam itself. Set them on the real webcam with a tool like
[cameractrls](https://github.com/soyersoyer/cameractrls) or `guvcview`. They show up
in the virtual camera right away, even mid-call.

## Limitations

- **Apps can't negotiate the format.** A v4l2loopback virtual camera offers one format,
  the one picked in the menu. Apps can't choose another resolution from their own
  settings, and controls changed through the virtual camera are not passed on to the
  real webcam.

- **Edges are not perfect.** Hair, fast hand movements and objects with the same
  colour as the background (e.g. a white headset in front of a white wall) can be
  partly blurred. This is fine for blurring. For a clean green-screen-style cutout,
  use a physical green screen.
- **OBS's own virtual camera** loads v4l2loopback with its own options. This package
  sets `devices=0` (no static devices), so OBS's virtual camera may need its module
  options adjusted.
- Only tested on AMD (RX 9070, MIGraphX) and CPU so far. NVIDIA/CUDA reports are
  welcome.

## License

GPL-3.0-or-later. The Robust Video Matting models are GPL-3.0 and are downloaded at
build/first run, not bundled in this repository.
