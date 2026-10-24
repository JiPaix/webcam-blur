# Webcam Blur

[![tests](https://github.com/JiPaix/webcam-blur/actions/workflows/tests.yml/badge.svg)](https://github.com/JiPaix/webcam-blur/actions/workflows/tests.yml)

**Blur the background of your webcam on Linux, in every app:** Zoom, Teams, Google
Meet, Discord, OBS, Firefox, Chrome… Turn it on and off with one click in the tray.

| No blur | Light | Medium | Strong |
| :---: | :---: | :---: | :---: |
| ![No blur](docs/images/strength-off.jpg) | ![Light blur](docs/images/strength-light.jpg) | ![Medium blur](docs/images/strength-medium.jpg) | ![Strong blur](docs/images/strength-strong.jpg) |

<sub>Photo: NASA portrait of astronaut Sally Ride (public domain).</sub>

## Why Webcam Blur?

- **Works everywhere.** Each webcam gets a twin called *"your webcam (blur)"*. Choose
  it in your app's camera settings, like any other camera. Your app doesn't need to
  support background blur.
- **Plug and play.** New webcams are picked up as soon as you plug them in. Nothing to
  configure.
- **One click.** Left-click the tray icon to switch the blur on or off for all your
  webcams. Right-click for per-webcam settings.
- **Stays out of the way.** Until an app actually uses a camera, it needs about 40 MB of
  memory and next to no processor time, and the webcam stays off.
- **Uses your graphics card** (AMD or NVIDIA) when it can, which takes most of the work
  off your processor. It falls back to the processor otherwise.
- **Private.** Everything happens on your computer. No account, no cloud, nothing sent
  anywhere.
- **In your language.** English and French so far. [Adding a language](CONTRIBUTING.md)
  only takes a text file.

## The tray icon

| <picture><source media="(prefers-color-scheme: dark)" srcset="docs/images/icon-on-dark.png"><img src="docs/images/icon-on-light.png" alt="Blur on" width="88"></picture> | <picture><source media="(prefers-color-scheme: dark)" srcset="docs/images/icon-live-dark.png"><img src="docs/images/icon-live-light.png" alt="Blur on, camera in use" width="88"></picture> | <picture><source media="(prefers-color-scheme: dark)" srcset="docs/images/icon-off-dark.png"><img src="docs/images/icon-off-light.png" alt="Blur off" width="88"></picture> |
| :---: | :---: | :---: |
| Blur on | A camera is in use | Blur off |

Left-click the icon to turn the blur on or off for all webcams. Right-click opens the
menu.

## The tray menu

![The Webcam Blur tray menu](docs/images/tray-menu.png)

- **One line per webcam** shows whether it's blurred and in use. It opens:
  - **Blur:** on or off for that webcam.
  - **Resolution:** every resolution and frame rate your webcam supports, read from the
    webcam itself (e.g. 1080p · 30 fps or 720p · 60 fps). *Automatic* picks 720p at
    30 fps, which is what video-call apps send anyway.
- **Blur strength:** light, medium or strong (see the pictures above).
- **Use GPU** and **Quality:** shown only if your graphics card passed a quick test at
  startup. *Standard* uses less memory. *High* gives slightly finer edges around hair.
- **Start at login.**
- A greyed-out line shows the processor, memory and graphics memory the app uses right now.

## Getting started

1. **Install it.** On Arch: `yay -S webcam-blur`. Other systems: see
   [Installation](#installation) below.
2. **Start "Webcam Blur"** from your applications menu. A webcam icon appears in the
   tray. Tick *Start at login* to have it every time.
3. **In your video app, select the camera ending in "(blur)"** instead of your
   webcam. That's it.

The icon's lens turns blue when the blur is on, and a red dot appears while an app is
using a camera.

## Questions

**Which apps does it work with?**
Any app that uses a webcam: Zoom, Teams, Meet and other browser-based calls, Discord,
Slack, Skype, OBS, Jitsi… If your app lists cameras, the "(blur)" ones are there.

**Will it slow down my computer?**
Only while an app uses a camera. On the test machine (Ryzen 7 9800X3D, Radeon RX 9070),
blurring a 720p webcam at 60 fps takes about 1.6 processor cores and 1.3 GB of memory
with the graphics card. Without the card it takes about 4 cores and 250 MB. When no app
uses a camera, it needs about 40 MB and next to no processor time. See
[Memory and speed](#memory-and-speed) for details.

**The "Use GPU" option is missing.**
Your graphics card didn't pass the startup test, or onnxruntime was installed without
GPU support. Blurring still works, on the processor. The reason is in the log:
`journalctl --user -t webcam-blur`. On Arch, install `python-onnxruntime-rocm` (AMD) or
`python-onnxruntime-cuda` (NVIDIA).

**The first call after installing takes a minute to use the GPU.**
That's normal on AMD. The model is prepared for your card once per resolution
(1–2 minutes, also after a graphics driver update), and the processor does the work
meanwhile.

**How do I change brightness, focus or zoom?**
Those settings live in the webcam itself. Change them with a tool like
[cameractrls](https://github.com/soyersoyer/cameractrls) or `guvcview`. The blurred
camera shows the change straight away, even during a call.

**Can my video app pick the resolution?**
No. Choose it in Webcam Blur's menu instead, under the webcam's *Resolution*. A new
resolution applies once no app is using the camera.

**Bits of my hair or my headphones get blurred.**
The person is detected by an AI model. It struggles with things the same colour as
the wall behind them (a white headset against a white wall, for example), with
flyaway hair and with fast movements. A background that contrasts with you, and light
on your face rather than behind you, help a lot. *High* quality helps a bit.

## Installation

### Arch Linux and derivatives

Webcam Blur is in the [AUR](https://aur.archlinux.org/packages/webcam-blur):

```sh
yay -S webcam-blur      # or paru -S webcam-blur, or any AUR helper
```

Without an AUR helper:

```sh
git clone https://aur.archlinux.org/webcam-blur.git
cd webcam-blur
makepkg -si
```

pacman asks which onnxruntime build to install: `python-onnxruntime-rocm` for an AMD
graphics card, `python-onnxruntime-cuda` for NVIDIA, otherwise `python-onnxruntime-cpu`.
You can switch later, e.g. `sudo pacman -S python-onnxruntime-rocm`.

The `v4l2loopback` kernel module (**version 0.13 or newer**) ships with some kernels,
such as CachyOS. For other kernels, install `v4l2loopback-dkms`.

### Other distributions / from source

Requirements: Python ≥ 3.11, PyQt6, OpenCV, NumPy, pyudev, onnx, onnxruntime (a GPU
build such as `onnxruntime-gpu` for NVIDIA, or ROCm/MIGraphX for AMD), and the
v4l2loopback module (≥ 0.13). Ubuntu 24.04 only packages v4l2loopback 0.12, which
cannot create virtual cameras at runtime: build a newer one from source there.

1. Install the system files (as root):

   ```sh
   install -Dm644 data/60-webcam-blur.rules /etc/udev/rules.d/60-webcam-blur.rules
   install -Dm644 data/modprobe.conf /etc/modprobe.d/webcam-blur.conf
   install -Dm644 data/modules-load.conf /etc/modules-load.d/webcam-blur.conf
   udevadm control --reload && modprobe v4l2loopback
   ```

2. Run the app. It downloads the models to `~/.local/share/webcam-blur/models` on
   first start:

   ```sh
   python -m webcam_blur
   ```

---

## Technical details

### How it works

```
              ┌──────────── tray process (always running, ~40 MB) ───────────────┐
 udev ──────▶ │ detects webcams, creates one v4l2loopback device per webcam,     │
              │ keeps it ready (format set, black frames), shows the menu        │
              └──────────┬───────────────────────────────────────────────────────┘
                         │ an app opens "<webcam> (blur)"
                         ▼
              ┌──────────── stream process (only while an app reads) ────────────┐
 webcam ────▶ │ capture → person matting (RVM) → blurred background → writes     │ ──▶ the app
              │ through the virtual camera's file descriptor handed over by the  │
              │ tray; GPU inference in its own subprocess                        │
              └──────────────────────────────────────────────────────────────────┘
```

- **Virtual cameras** are [v4l2loopback](https://github.com/v4l2loopback/v4l2loopback)
  devices, created and removed at runtime through `/dev/v4l2loopback`. A udev rule
  gives the logged-in user access to it. The module is loaded with `devices=0`.
- **When a webcam is unplugged,** its virtual camera is removed. If an app still holds
  it open, removal waits until the app lets go. Greyscale-only devices, such as IR
  face-login cameras, are skipped.
- **The tray process never loads OpenCV, onnxruntime or a model.** When an app opens a
  virtual camera, the tray starts the stream process and hands it the virtual camera's
  already-configured file descriptor, because the format can't change under a reading
  app. Settings changes go to the stream process over stdin, and it reports its state
  on stdout. It exits a few seconds after the app closes the camera.
- **People are cut out** with [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting)
  (ONNX Runtime). The background is blurred with a background-weighted Gaussian, so the
  person doesn't bleed into the blur.
- **On the processor,** MobileNetV3 runs on a frame scaled down to 640 px.
- **On the graphics card,** the model runs at full resolution: MobileNetV3 for
  *Standard*, ResNet50 for *High*, in half precision on MIGraphX. GPU execution
  providers need fixed input sizes, so the model is converted once per resolution
  (`~/.cache/webcam-blur/`). MIGraphX also compiles it once per resolution.
- **The GPU check** runs a tiny model on the GPU in a throwaway process at startup. A
  provider that silently falls back to the processor counts as a failure.

### Memory and speed

Measured on a Ryzen 7 9800X3D with a Radeon RX 9070 (ROCm 7). Webcam at 1280×720 and
60 fps, 59.5 fps delivered in every mode. Memory is the PSS of all the app's processes:

| | Memory | Processor | Graphics memory |
| --- | --- | --- | --- |
| No app using a camera | 39 MB | ~2 % of one core | — |
| GPU, *Standard* quality | 1.3 GB | 1.6 cores | 616 MB |
| GPU, *High* quality | 1.7 GB | 1.6 cores | 678 MB |
| Processor only | 250 MB | 3.9 cores | — |

ROCm and MIGraphX alone take about 470 MB as soon as anything runs on an AMD GPU. That
is why the processor path uses much less memory. With the GPU, the processor still
decodes the webcam's video, composes the blur and converts formats. At 30 fps,
processor use roughly halves (0.8 cores measured with *Standard*). The first call at a new resolution also compiles the
model, which temporarily uses more memory.

### Configuration file

Everything is in the tray menu. Settings are stored in
`~/.config/webcam-blur/config.json`:

| Key | Meaning |
| --- | --- |
| `strength` | `light`, `medium` or `strong` |
| `use_gpu` | `true` / `false` |
| `quality` | `standard` or `high` (GPU model) |
| `language` | e.g. `"fr"`; `null` follows the system locale |
| `cameras.<id>.blur` | blur on/off for this camera |
| `cameras.<id>.mode` | resolution chosen in the menu (`{"width", "height", "fps"}`), `null` = automatic |
| `cameras.<id>.virtual` | `false` to never create a virtual camera for this webcam |

### Limitations

- **Apps can't negotiate the format.** A v4l2loopback virtual camera offers one format,
  the one picked in the menu. Controls changed through the virtual camera are not
  passed on to the real webcam.
- **OBS's own virtual camera** expects to load v4l2loopback with its own options. With
  `devices=0`, it may need its module options adjusted.
- **Tested hardware:** only AMD (RX 9070, MIGraphX) and processors so far. NVIDIA/CUDA
  reports are welcome.

### Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for translations, the code layout and the test
suite. CI runs the unit tests on Python 3.11–3.14, end-to-end tests against the real
v4l2loopback kernel module (a fake webcam fed to the real app logic), and builds and
tests the Arch package.

## License

GPL-3.0-or-later. The Robust Video Matting models are GPL-3.0. They are downloaded at
build time or on first run, not stored in this repository.
