"""airspace entry point.

    python -m airspace [--fullscreen] [--no-camera] [--raster] [--reset] [--verbose]
"""

from __future__ import annotations

import sys


def main() -> int:
    from airspace.app.application import run

    return run(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
