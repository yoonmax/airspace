"""Mapping between camera-normalised coordinates and view (widget) coordinates.

Conventions
-----------
* Tracking runs on the *display* frame, which is already mirrored when the
  mirror option is on. Normalised coordinates ``(u, v)`` are therefore in
  display space: ``u = 0`` is the left edge of what the user sees.
* The camera image is drawn with "cover" scaling: it fills the view while
  preserving aspect ratio, cropping whichever dimension overflows.
* ``reach`` expands pointer motion around the centre of the visible region so
  users can reach screen edges without stretching. ``reach == 1`` keeps the
  cursor exactly on the fingertip.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ViewRect:
    x: float
    y: float
    w: float
    h: float


def mirror_u(u: float) -> float:
    return 1.0 - u


class CoordinateMapper:
    def __init__(self, frame_w: int = 1280, frame_h: int = 720, view_w: float = 1280,
                 view_h: float = 720, reach: float = 1.0, edge_margin: float = 2.0) -> None:
        self.frame_w = max(1, int(frame_w))
        self.frame_h = max(1, int(frame_h))
        self.view_w = max(1.0, float(view_w))
        self.view_h = max(1.0, float(view_h))
        self.reach = reach
        self.edge_margin = edge_margin
        self._update()

    def configure(self, frame_w: int | None = None, frame_h: int | None = None,
                  view_w: float | None = None, view_h: float | None = None,
                  reach: float | None = None) -> None:
        if frame_w:
            self.frame_w = int(frame_w)
        if frame_h:
            self.frame_h = int(frame_h)
        if view_w:
            self.view_w = float(view_w)
        if view_h:
            self.view_h = float(view_h)
        if reach is not None:
            self.reach = float(reach)
        self._update()

    def _update(self) -> None:
        scale = max(self.view_w / self.frame_w, self.view_h / self.frame_h)
        w, h = self.frame_w * scale, self.frame_h * scale
        self.scale = scale
        self.image_rect = ViewRect((self.view_w - w) / 2.0, (self.view_h - h) / 2.0, w, h)
        r = self.image_rect
        # Normalised region of the frame that is actually visible.
        self.visible_u = (-r.x / r.w, (self.view_w - r.x) / r.w)
        self.visible_v = (-r.y / r.h, (self.view_h - r.y) / r.h)

    # -------------------------------------------------------------- direct
    def norm_to_view(self, u: float, v: float) -> tuple[float, float]:
        """Where a point of the camera image appears on screen (overlays)."""
        r = self.image_rect
        return r.x + u * r.w, r.y + v * r.h

    def view_to_norm(self, x: float, y: float) -> tuple[float, float]:
        r = self.image_rect
        return (x - r.x) / r.w, (y - r.y) / r.h

    def norm_length_to_view(self, length: float) -> float:
        """Convert a length measured in normalised *width* units to pixels."""
        return length * self.image_rect.w

    # ------------------------------------------------------------- pointer
    def pointer_to_view(self, u: float, v: float, clamp: bool = True) -> tuple[float, float]:
        """Map a fingertip to the cursor position, applying reach and edge clamping."""
        cu = (self.visible_u[0] + self.visible_u[1]) / 2.0
        cv = (self.visible_v[0] + self.visible_v[1]) / 2.0
        u2 = cu + (u - cu) * self.reach
        v2 = cv + (v - cv) * self.reach
        x, y = self.norm_to_view(u2, v2)
        if clamp:
            m = self.edge_margin
            x = min(max(x, m), self.view_w - m)
            y = min(max(y, m), self.view_h - m)
        return x, y

    def view_to_pointer(self, x: float, y: float) -> tuple[float, float]:
        """Inverse of :meth:`pointer_to_view` (without clamping)."""
        u2, v2 = self.view_to_norm(x, y)
        cu = (self.visible_u[0] + self.visible_u[1]) / 2.0
        cv = (self.visible_v[0] + self.visible_v[1]) / 2.0
        return cu + (u2 - cu) / self.reach, cv + (v2 - cv) / self.reach

    def interaction_region(self) -> ViewRect:
        """The camera-space rectangle (in view coordinates) that covers the whole view."""
        u0, v0 = self.view_to_pointer(0, 0)
        u1, v1 = self.view_to_pointer(self.view_w, self.view_h)
        x0, y0 = self.norm_to_view(u0, v0)
        x1, y1 = self.norm_to_view(u1, v1)
        return ViewRect(x0, y0, x1 - x0, y1 - y0)
