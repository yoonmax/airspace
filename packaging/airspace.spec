# PyInstaller spec for airspace.
#   macOS:   pyinstaller packaging/airspace.spec   →  dist/airspace.app
#   Windows/Linux: same command → dist/airspace/ (one-folder build)
# Run `python scripts/download_models.py` first so the models are bundled.
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).parent
models = ROOT / "airspace" / "assets" / "models"
if not (models / "hand_landmarker.task").exists():
    raise SystemExit("Models missing: run `python scripts/download_models.py` before building.")

datas = [(str(ROOT / "airspace" / "assets"), "airspace/assets")]
datas += collect_data_files("mediapipe")
datas += collect_data_files("trimesh")

# License texts must travel with the binaries: the summary, the GNU texts for
# Qt/PySide6 (LGPL v3, which builds on GPL v3) and FFmpeg (LGPL v2.1), and each
# bundled package's own license files from its installed metadata.
datas += [(str(ROOT / "THIRD_PARTY_NOTICES.md"), "."), (str(ROOT / "licenses"), "licenses")]
_py_license = Path(sys.base_prefix) / "lib" / f"python{sys.version_info[0]}.{sys.version_info[1]}" / "LICENSE.txt"
if _py_license.exists():  # the bundled Python runtime (PSF License)
    datas.append((str(_py_license), "licenses/python"))
from importlib.metadata import PackageNotFoundError, distribution  # noqa: E402
from pathlib import PurePosixPath  # noqa: E402

for _name in ("mediapipe", "opencv-contrib-python", "numpy", "psutil", "moderngl", "glcontext", "trimesh",
              "pillow", "PySide6", "PySide6_Essentials", "PySide6_Addons", "shiboken6"):
    try:
        _dist = distribution(_name)
    except PackageNotFoundError:
        continue
    for _f in _dist.files or []:
        _parts = PurePosixPath(str(_f)).parts
        if _parts and _parts[0].endswith(".dist-info") and any(
                k in _parts[-1].lower() for k in ("license", "copying", "notice")):
            _dest = PurePosixPath("licenses", _name, *_parts[1:-1])
            datas.append((str(_f.locate()), str(_dest)))
binaries = collect_dynamic_libs("mediapipe")
hiddenimports = (collect_submodules("airspace.miniapps") + ["glcontext", "moderngl", "PySide6.QtOpenGLWidgets",
                                                              "PySide6.QtMultimedia"])

a = Analysis(
    [str(ROOT / "airspace" / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="airspace", console=False,
          argv_emulation=False, target_arch=None, codesign_identity=None, entitlements_file=None)
coll = COLLECT(exe, a.binaries, a.datas, name="airspace")

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="airspace.app",
        bundle_identifier="app.airspace.desktop",
        info_plist={
            "CFBundleName": "airspace",
            "CFBundleDisplayName": "airspace",
            "CFBundleShortVersionString": "0.1.0",
            "NSCameraUsageDescription": "airspace uses your camera on this computer to track your hands. "
                                        "Video never leaves your device.",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.0",
        },
    )
