"""Download the MediaPipe model files used by airspace.

Models are published by Google under the Apache 2.0 license:
  https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
  https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter

Usage:
    python scripts/download_models.py [--force]
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from airspace.vision.model_files import MODELS, download_model, model_path  # noqa: E402


def main() -> int:
    force = "--force" in sys.argv
    print(f"airspace model setup → {model_path('').parent}")
    try:
        for model in MODELS:
            target = model_path(model.name)
            if target.exists() and target.stat().st_size > 0 and not force:
                print(f"  ✓ {model.name} already present ({target.stat().st_size / 1e6:.2f} MB)")
                continue
            download_model(model, lambda msg: print("  ↓", msg))
            digest = hashlib.sha256(target.read_bytes()).hexdigest()[:16]
            print(f"    saved {target.stat().st_size / 1e6:.2f} MB  sha256:{digest}…")
    except Exception as exc:  # noqa: BLE001 - user-facing script
        print(f"\nDownload failed: {exc}\nCheck your internet connection and retry.")
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
