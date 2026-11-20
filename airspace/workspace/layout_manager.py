"""Intelligent window placement.

New windows are placed where they cover the least of the user (especially
the face), stay out of the active hand workspace, and avoid existing windows,
preferring the edges of the screen. Placement only happens when a window is
opened or on an explicit "arrange" — windows never wander on their own.
"""

from __future__ import annotations

import numpy as np

from airspace.workspace.geometry import Rect

GRID_W, GRID_H = 64, 36


class PlacementModel:
    def __init__(self) -> None:
        self.person_cost: np.ndarray | None = None  # (GRID_H, GRID_W) in view space

    def update_person(self, person_view_grid: np.ndarray | None) -> None:
        """Set the person-occupancy grid (values 0..1), already in view space."""
        if person_view_grid is None:
            self.person_cost = None
            return
        g = np.clip(person_view_grid.astype(np.float64), 0, 1)
        rows = np.nonzero(g.max(axis=1) > 0.5)[0]
        weight = np.ones_like(g)
        if rows.size:
            top = rows[0]
            span = max(1, rows[-1] - top)
            face_rows = slice(top, min(GRID_H, top + max(2, int(span * 0.38))))
            weight[face_rows, :] = 3.0  # the face matters most
        self.person_cost = g * weight

    def cost_grid(self, bounds: Rect, top_inset: float, windows: list[Rect]) -> np.ndarray:
        cost = np.zeros((GRID_H, GRID_W), np.float64)
        if self.person_cost is not None:
            cost += self.person_cost * 2.2
        # Active hand workspace: the lower-central area where hands usually are.
        yy, xx = np.mgrid[0:GRID_H, 0:GRID_W]
        u, v = (xx + 0.5) / GRID_W, (yy + 0.5) / GRID_H
        cost += 0.45 * np.exp(-(((u - 0.5) / 0.22) ** 2 + ((v - 0.72) / 0.25) ** 2))
        # Mild preference for edges over the centre.
        cost += 0.25 * np.exp(-(((u - 0.5) / 0.3) ** 2))
        # Existing windows.
        for r in windows:
            x0 = int(np.clip((r.x - bounds.x) / bounds.w * GRID_W, 0, GRID_W))
            x1 = int(np.clip((r.right - bounds.x) / bounds.w * GRID_W, 0, GRID_W))
            y0 = int(np.clip((r.y - bounds.y) / bounds.h * GRID_H, 0, GRID_H))
            y1 = int(np.clip((r.bottom - bounds.y) / bounds.h * GRID_H, 0, GRID_H))
            cost[y0:y1, x0:x1] += 3.0
        return cost

    def place(self, w: float, h: float, bounds: Rect, top_inset: float, bottom_inset: float,
              windows: list[Rect], margin: float = 24.0) -> Rect:
        cost = self.cost_grid(bounds, top_inset, windows)
        integral = np.pad(cost, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        cw, ch = bounds.w / GRID_W, bounds.h / GRID_H
        gw = max(1, int(round(w / cw)))
        gh = max(1, int(round(h / ch)))
        best, best_cost = None, float("inf")
        x_min = bounds.x + margin
        x_max = bounds.right - w - margin
        y_min = bounds.y + top_inset + 8
        y_max = bounds.bottom - h - bottom_inset - margin
        if x_max < x_min:
            x_max = x_min = bounds.x + max(0.0, (bounds.w - w) / 2)
        if y_max < y_min:
            y_max = y_min
        for y in np.linspace(y_min, y_max, 12):
            for x in np.linspace(x_min, x_max, 20):
                gx = int(np.clip((x - bounds.x) / cw, 0, GRID_W - 1))
                gy = int(np.clip((y - bounds.y) / ch, 0, GRID_H - 1))
                gx1, gy1 = min(GRID_W, gx + gw), min(GRID_H, gy + gh)
                area_cost = integral[gy1, gx1] - integral[gy, gx1] - integral[gy1, gx] + integral[gy, gx]
                c = area_cost / max(1, gw * gh)
                if c < best_cost - 1e-9:
                    best, best_cost = (x, y), c
        assert best is not None
        return Rect(float(best[0]), float(best[1]), w, h)

    def arrange(self, sizes: list[tuple[float, float]], bounds: Rect, top_inset: float,
                bottom_inset: float) -> list[Rect]:
        """Greedy arrangement: largest windows first, each in the best remaining spot."""
        order = sorted(range(len(sizes)), key=lambda i: -sizes[i][0] * sizes[i][1])
        placed: dict[int, Rect] = {}
        for i in order:
            w, h = sizes[i]
            placed[i] = self.place(w, h, bounds, top_inset, bottom_inset, list(placed.values()))
        return [placed[i] for i in range(len(sizes))]
