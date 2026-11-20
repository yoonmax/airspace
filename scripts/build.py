"""Reproducible desktop build.

    python scripts/build.py          # downloads models if needed, runs PyInstaller

Output: dist/airspace.app (macOS) or dist/airspace/ (Windows / Linux).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "download_models.py")])
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", str(ROOT / "packaging" / "airspace.spec")]
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)
    out = ROOT / "dist" / ("airspace.app" if sys.platform == "darwin" else "airspace")
    print(f"\nBuilt {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
