# Third-party notices

airspace depends on the following third-party software and models. Each is
used under its own license; this file summarises them for convenience. The
authoritative license texts ship with each package (see the package's
`*.dist-info/` folder in your environment) or at the linked sources.

| Component | Used for | License |
|---|---|---|
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) | hand landmark detection, person segmentation runtime | Apache License 2.0 |
| MediaPipe **Hand Landmarker** model (`hand_landmarker.task`) | 21-point hand landmarks | Apache License 2.0 (Google) |
| MediaPipe **Selfie Segmenter** model (`selfie_segmenter.tflite`) | person mask for occlusion & window placement | Apache License 2.0 (Google) |
| [Qt for Python / PySide6](https://www.qt.io/qt-for-python) | desktop UI, rendering, multimedia | LGPL v3 (Qt also offers commercial licenses) |
| Qt Multimedia FFmpeg backend | audio decoding | LGPL v2.1+ |
| [OpenCV](https://opencv.org) (`opencv-contrib-python`) | webcam capture, image processing | Apache License 2.0 |
| [NumPy](https://numpy.org) | numerics | BSD 3-Clause |
| [psutil](https://github.com/giampaolo/psutil) | system monitor | BSD 3-Clause |
| [ModernGL](https://github.com/moderngl/moderngl) / glcontext | 3D viewer rendering | MIT |
| [trimesh](https://github.com/mikedh/trimesh) | GLB/GLTF/OBJ/STL import | MIT |
| [Pillow](https://python-pillow.org) | image support | MIT-CMU (HPND) |
| [Python](https://www.python.org) runtime (bundled in packaged builds) | interpreter | PSF License |
| [PyInstaller](https://pyinstaller.org) (build only) | packaging | GPL v2 with bootloader exception |

## Bundled content

* The model files are **not** committed to the repository; `scripts/download_models.py`
  (or the in-app prompt) downloads them from Google's official model storage.
* All 3D models, sample images, the demo audio track and interface sounds are
  generated procedurally by airspace's own code at runtime. No third-party
  artwork, music or fonts are bundled. The interface uses fonts already
  installed on the system.

## Packaged builds

When distributing a PyInstaller build, the LGPL obligations of Qt/PySide6
apply: PySide6/Qt libraries are shipped as separate dynamic libraries inside
the bundle so they can be replaced, and their license texts accompany the
distribution. The packaged app contains this file and a `licenses/` folder
(`airspace.app/Contents/Resources/licenses/`) with the GNU LGPL v3, GPL v3 and
LGPL v2.1 texts (from gnu.org, also in this repository's `licenses/` folder)
and the license files of every bundled Python package. Qt and FFmpeg sources
are available from their upstream projects (qt.io, ffmpeg.org).
