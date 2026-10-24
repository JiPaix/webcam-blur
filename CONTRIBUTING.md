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

- `webcam_blur/app.py`: tray icon and menu, matching virtual cameras to webcams.
- `webcam_blur/worker.py`: one thread per webcam (capture → blur → virtual camera).
- `webcam_blur/matting.py`: RVM models (CPU, and the GPU subprocess).
- `webcam_blur/v4l2.py`: kernel interfaces (v4l2loopback control, output, webcam modes).
- `webcam_blur/cameras.py`: webcam detection and hotplug (udev).

Run from a checkout with `python -m webcam_blur` (see the README for the system files).
