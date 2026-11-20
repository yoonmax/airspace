"""Hit-testing helpers shared by the workspace layers."""

from __future__ import annotations

import math


def radial_index(x: float, y: float, cx: float, cy: float, count: int, inner: float, outer: float,
                 start_angle: float = -math.pi / 2) -> int | None:
    """Sector index under (x, y) for a radial menu, or None in the hub/outside.

    Sectors are generous on purpose: anywhere in the ring between ``inner``
    and ``outer`` selects the nearest item by angle, which is far more
    forgiving for hand input than requiring the pointer inside a small icon.
    """
    if count <= 0:
        return None
    dx, dy = x - cx, y - cy
    r = math.hypot(dx, dy)
    if r < inner or r > outer:
        return None
    a = (math.atan2(dy, dx) - start_angle) % (2 * math.pi)
    step = 2 * math.pi / count
    return int((a + step / 2) // step) % count


def item_angle(index: int, count: int, start_angle: float = -math.pi / 2) -> float:
    return start_angle + index * 2 * math.pi / count
