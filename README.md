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
  webcams. Right-click for the settings.
- **Your webcam's settings, kept.** Brightness, exposure, white balance, focus and zoom
  as sliders, remembered and set again whenever the webcam is plugged in.
- **Stays out of the way.** Until an app actually uses a camera, it needs about 40 MB of
  memory and next to no processor time, and the webcam stays off.
- **Light by default.** About 0.6 of a processor core and 220 MB of memory at 720p and
  30 fps. A graphics card is optional (AMD, NVIDIA or Intel, through Vulkan): for about
  50 MB more memory, it takes the load off the processor at higher qualities,
  resolutions and frame rates.
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

- **Webcam settings…** opens the settings window (below).
- **Blur strength:** light, medium or strong (see the pictures above).
- **Quality:** how finely the AI model looks at you. *Low*, the default, is the
  lightest, and its edges differ very little from *Standard*. *High* gives slightly
  finer edges around hair; it is much heavier, best with a graphics card. All the
  numbers are in [Memory and speed](#memory-and-speed).
- **Use GPU:** off by default. Shown only if your graphics card passed a quick test at
  startup. It saves the most processor time with *Standard* and *High*, or at 60 fps.
- **Start at login.**
- A greyed-out line shows the processor, memory and graphics memory the app uses right now.

## The settings window

<img src="docs/images/settings.png" alt="The webcam settings window" width="420">

One tab per webcam:

- **Blur the background:** on or off for that webcam. The tab also shows whether an app
  is using it.
- **Resolution:** every resolution and frame rate your webcam supports, read from the
  webcam itself (e.g. 1080p · 30 fps or 720p · 60 fps). *Automatic* picks 720p at
  30 fps, which is what video-call apps send anyway.
- **Image:** the webcam's own controls, as it lists them: brightness, contrast, gain,
  white balance, exposure, anti-flicker, focus, zoom… Changes show straight away, even
  during a call. Settings the webcam ignores in its current mode are greyed out, such as
  the exposure time while exposure is automatic. Under the exposure time, a note says how
  long it can be before the webcam has to lower its frame rate.
- **Reset image settings** puts the webcam's defaults back.

The image settings are stored in the webcam, so they also apply to the camera without
blur. Webcams forget them when unplugged: Webcam Blur remembers the ones you changed and
sets them again whenever the webcam is plugged in or the app starts.

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
Only while an app uses a camera. With the default settings (720p at 30 fps, *Low*
quality, on the processor), blurring takes about 0.6 of a processor core and 220 MB of
memory on the test machine (Ryzen 7 9800X3D), or a full core in 1080p. When no app uses a
camera, it needs about 40 MB and next to no processor time. See
[Memory and speed](#memory-and-speed) for every setting.

**My computer is short on memory or power.**
Keep *Low* quality. If *Use GPU* is offered, ticking it saves processor time for about
50 MB more memory. If the processor can't keep up (with *High*, for example), the video
keeps its frame rate and the cut-out simply updates less often.

**The "Use GPU" option is missing.**
The GPU runs the model through Vulkan with [ncnn](https://github.com/Tencent/ncnn): on
Arch, `sudo pacman -S ncnn`, then restart Webcam Blur. It works with the usual Vulkan
drivers (`vulkan-radeon` for AMD, `vulkan-intel` for Intel, the NVIDIA driver). At
startup, a quick test runs the model on the GPU and on the processor: the option only
appears if the GPU gives the same result, faster. Blurring still works without it, on
the processor. The reason is in the log: `journalctl --user -t webcam-blur`.

**I chose 60 fps but my app gets 30.**
Many webcams lower their frame rate in dim light to expose each picture longer. On
Logitech webcams this is *Lower the frame rate in dim light*, under *Exposure* in the
settings window: untick it. The picture gets darker instead. More light on your face
fixes both.

**How do I change brightness, focus or zoom?**
In *Webcam settings…*, under *Image*. The blurred camera shows the change straight
away, even during a call.

**Can my video app pick the resolution?**
No. Choose it in *Webcam settings…*, under *Resolution*. A new resolution applies once
no app is using the camera.

**Bits of my hair or my headphones get blurred.**
The person is detected by an AI model. It struggles with things the same colour as
the wall behind them (a white headset against a white wall, for example), with
flyaway hair and with fast movements. A background that contrasts with you, and light
on your face rather than behind you, help a lot. *High* quality helps a bit, and
*Standard* can help with fast movements (*Low* updates the cut-out 30 times a second).

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

For GPU acceleration (optional), also install `ncnn`. If pacman asks which onnxruntime
build to install, pick `python-onnxruntime-cpu`: the GPU doesn't go through onnxruntime.

The `v4l2loopback` kernel module (**version 0.13 or newer**) ships with some kernels,
such as CachyOS. For other kernels, install `v4l2loopback-dkms`.

### Other distributions / from source

Requirements: Python ≥ 3.11, PyQt6, OpenCV, NumPy, pyudev, onnxruntime, and the
v4l2loopback module (≥ 0.13). For the GPU: the ncnn library (`libncnn.so.1`, built with
Vulkan) and a Vulkan driver. Ubuntu 24.04 only packages v4l2loopback 0.12, which
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
              │ tray; the model runs on the CPU or on the GPU (Vulkan)           │
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
- **People are cut out** with [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting):
  ONNX Runtime on the processor, [ncnn](https://github.com/Tencent/ncnn) with Vulkan on the
  graphics card. The background is blurred with a background-weighted Gaussian, so the
  person doesn't bleed into the blur.
- **Quality** sets the model and the size of the frame it sees. The cut-out is then
  scaled back to the webcam's resolution:

  | | Processor | Graphics card |
  | --- | --- | --- |
  | *Low* | MobileNetV3, 320 px wide | MobileNetV3, 640 px wide |
  | *Standard* | MobileNetV3, 640 px wide | MobileNetV3, full resolution |
  | *High* | ResNet50, 640 px wide | ResNet50, full resolution |

  Above 30 fps, *Low* runs the model on every other frame and reuses the cut-out in
  between. Whatever the quality, when the model takes longer than a frame lasts (the
  stream measures how busy it is every second), it runs on every second, third…
  frame instead, so the video keeps the webcam's frame rate.
- **On the graphics card,** the models run in half precision on ncnn's Vulkan backend,
  called from Python through ncnn's C API. They are converted once from RVM's ONNX files
  by [`tools/convert_ncnn.py`](tools/convert_ncnn.py) and published as a
  [release](https://github.com/JiPaix/webcam-blur/releases/tag/ncnn-models-1). The
  weights work at any size: the network description is generated for each webcam
  resolution from a 640×360 template, in a fraction of a second, with nothing to compile.
  The model loads in the background while the processor covers the first frames.
- **The GPU check** runs the model on a portrait at startup, in a throwaway process, on
  the GPU and on the processor. The GPU must give the same cut-out (a driver bug would
  show as wrong edges: one in ncnn's Vulkan code is already worked around in the
  conversion) and be faster (a software Vulkan driver is not).

### Memory and speed

Measured on a Ryzen 7 9800X3D with a Radeon RX 9070 XT (Vulkan, RADV) and a Logitech
C922, which films at 60 fps up to 720p and at 30 fps in 1080p. Every setting delivers
the webcam's full frame rate. Memory is the PSS of all the app's processes, in the same
units as the tray menu:

| | 720p, 30 fps | 720p, 60 fps | 1080p, 30 fps | Memory (720p / 1080p) | Graphics memory (720p / 1080p) |
| --- | --- | --- | --- | --- | --- |
| No app using a camera | 0.02 cores | 0.02 cores | 0.02 cores | 40 MB | — |
| Processor only, *Low* (default) | 0.6 cores | 0.9 cores | 1.0 cores | 220 / 240 MB | — |
| Processor only, *Standard* | 1.3 cores | 1.6 cores | 1.8 cores | 260 / 280 MB | — |
| Processor only, *High* | 1.7 cores | 2.5 cores¹ | 2.1 cores | 480 / 490 MB | — |
| GPU, *Low* | 0.6 cores | 0.8 cores | 0.9 cores | 260 / 290 MB | 60 / 60 MB |
| GPU, *Standard* | 0.6 cores | 1.1 cores | 1.0 cores | 280 / 350 MB | 80 / 110 MB |
| GPU, *High* | 0.6 cores | 1.1 cores | 1.0 cores | 320 / 360 MB | 200 / 230 MB |

¹ The model runs on every third frame there: on this processor, ResNet50 takes longer
than a 60 fps frame lasts.

With the GPU, the processor still decodes the webcam's video, composes the blur and
converts formats: that is most of what remains, and it grows with the resolution. At
30 fps, *Low* costs about the same either way; the GPU makes the difference at higher
qualities and frame rates.

### Configuration file

Everything is in the tray menu and the settings window. Settings are stored in
`~/.config/webcam-blur/config.json`:

| Key | Meaning |
| --- | --- |
| `strength` | `light`, `medium` or `strong` |
| `use_gpu` | `true` / `false` (default) |
| `quality` | `low` (default), `standard` or `high` |
| `language` | e.g. `"fr"`; `null` follows the system locale |
| `cameras.<id>.blur` | blur on/off for this camera |
| `cameras.<id>.mode` | resolution chosen in the settings (`{"width", "height", "fps"}`), `null` = automatic |
| `cameras.<id>.controls` | image settings changed in the settings window, e.g. `{"gain": 100}`, set again when the webcam is plugged in |
| `cameras.<id>.virtual` | `false` to never create a virtual camera for this webcam |

### Limitations

- **Apps can't negotiate the format.** A v4l2loopback virtual camera offers one format,
  the one picked in the menu. Controls changed through the virtual camera are not
  passed on to the real webcam.
- **OBS's own virtual camera** expects to load v4l2loopback with its own options. With
  `devices=0`, it may need its module options adjusted.
- **Tested hardware:** only AMD (RX 9070 XT, RADV) and processors so far. Reports from
  NVIDIA and Intel GPUs are welcome; the startup check keeps the GPU off if it misbehaves.

### Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for translations, the code layout and the test
suite. CI runs the unit tests on Python 3.11–3.14, end-to-end tests against the real
v4l2loopback kernel module (a fake webcam fed to the real app logic), and builds and
tests the Arch package.

## License

GPL-3.0-or-later. The Robust Video Matting models are GPL-3.0. They are downloaded at
build time or on first run, not stored in this repository.
