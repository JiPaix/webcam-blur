# Webcam Blur

Background blur for any webcam on Linux, usable in **every** app (Discord, Zoom,
Teams, Meet, OBS, browsers…).

For each webcam you plug in, Webcam Blur creates a virtual camera named
`<your webcam> (blur)`. Pick it in your app instead of the real webcam. A tray icon
turns the blur on and off per camera and sets its strength.

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

For GPU acceleration, also install `python-onnx` and the onnxruntime build for your GPU:

```sh
sudo pacman -S python-onnx python-onnxruntime-rocm   # AMD
sudo pacman -S python-onnx python-onnxruntime-cuda   # NVIDIA
```

Then start **Webcam Blur** from your app menu. Tick *Start at login* in its tray menu
to keep it running.

The `v4l2loopback` kernel module ships with some kernels (e.g. CachyOS). For other
kernels, install `v4l2loopback-dkms`.

### Other distributions / from source

Requirements: Python ≥ 3.11, PyQt6, OpenCV, NumPy, pyudev, onnxruntime (plus
`onnx` for GPU), and the v4l2loopback module.

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

Most settings are in the tray menu. They are stored in
`~/.config/webcam-blur/config.json`:

| Key | Meaning |
| --- | --- |
| `strength` | `light`, `medium` or `strong` |
| `use_gpu` | `true` / `false` |
| `language` | e.g. `"fr"`; `null` follows the system locale |
| `cameras.<id>.blur` | blur on/off for this camera |
| `cameras.<id>.virtual` | create a virtual camera for this webcam |
| `cameras.<id>.width` / `height` / `fps` | force a capture mode (default: 1280x720 if available) |

## Limitations

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
