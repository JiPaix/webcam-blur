# Contributing

Bug reports, translations and patches are welcome.

## Adding or improving a translation

Translations are plain JSON files in [`webcam_blur/locales/`](webcam_blur/locales).
No tools or build step needed.

1. Copy `en.json` to `<code>.json`, where `<code>` is the language code from your
   locale: `de` for German, `pt_BR` for Brazilian Portuguese, etc. A file named
   after the full locale (`pt_BR`) wins over the language alone (`pt`).
2. Translate the **values**, not the keys. Keep the `{placeholders}` as they are,
   and set `_language` to the language's own name (e.g. `"Deutsch"`).
3. Add your name to `_translators` if you like.
4. Test it:

   ```sh
   LANGUAGE=de python -m webcam_blur
   ```

5. Open a pull request.

Missing keys fall back to English, so a partial translation still works. When new
keys are added to `en.json`, other languages can catch up later.

One constraint: `virtual_camera.label` must stay short. The virtual camera name is
limited to 31 bytes, and long webcam names get shortened with `…` to make room for
the suffix.

## Code

- `webcam_blur/app.py`: tray icon and menu.
- `webcam_blur/settings.py`: the webcam settings window (blur, resolution, image controls).
- `webcam_blur/manager.py`: matches virtual cameras to plugged-in webcams.
- `webcam_blur/worker.py`: one thread per webcam: keeps its virtual camera ready and
  starts the stream process while an app reads it.
- `webcam_blur/stream.py`: the stream process (capture → matting → blur → virtual camera).
- `webcam_blur/matting.py`: RVM models (CPU, and the GPU subprocess).
- `webcam_blur/prepare.py`: freezes and compiles GPU models ahead of time, at low priority.
- `webcam_blur/models.py`: model files and the model per quality.
- `webcam_blur/gpucheck.py`: the startup GPU check, in a throwaway process.
- `webcam_blur/resources.py`: the resource line of the menu.
- `webcam_blur/v4l2.py`: kernel interfaces (v4l2loopback control, output, webcam modes).
- `webcam_blur/cameras.py`: webcam detection and hotplug (udev).

The tray process must stay light: it never imports OpenCV, onnxruntime, NumPy or onnx
(a test checks it). Anything that needs them runs in a subprocess.

Run from a checkout with `python -m webcam_blur` (see the README for the system files).

## Tests

```sh
pip install pytest   # plus the app's dependencies
python -m pytest -m "not v4l2"   # no kernel module needed
python -m pytest -m v4l2         # end-to-end: needs v4l2loopback >= 0.13 and access to /dev/v4l2loopback
```

The end-to-end tests feed a fake webcam (a loopback device playing a test portrait) to
the real app logic and check the virtual camera it produces. They create and remove
their own loopback devices and leave your real webcams alone. CI runs everything,
against v4l2loopback 0.15 and its development branch. It also builds and tests the Arch package.

## Releasing

1. Bump `__version__` in `webcam_blur/__init__.py` and `pkgver` in
   `packaging/arch/PKGBUILD`, and commit.
2. Tag and push: `git tag -a v0.1.2 -m v0.1.2 && git push origin main v0.1.2`.

The [release workflow](.github/workflows/release.yml) then:
- builds and checks the package from the release tarball;
- publishes it to the [AUR](https://aur.archlinux.org/packages/webcam-blur);
- creates the GitHub release;
- commits the tarball's checksum to `packaging/arch/PKGBUILD`.

It needs the repository secret `AUR_SSH_PRIVATE_KEY`: a private SSH key whose public
half is registered on the maintainer's AUR account.
