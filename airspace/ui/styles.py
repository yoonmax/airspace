"""Design tokens for airspace.

A restrained palette: deep charcoal glass, warm white type, a cool blue-grey
accent. High-contrast and large-button variants are derived from the base.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from PySide6.QtGui import QColor, QFont, QFontDatabase
from PySide6.QtWidgets import QApplication


def rgba(r: int, g: int, b: int, a: float = 1.0) -> QColor:
    c = QColor(r, g, b)
    c.setAlphaF(max(0.0, min(1.0, a)))
    return c


def with_alpha(c: QColor, a: float) -> QColor:
    out = QColor(c)
    out.setAlphaF(max(0.0, min(1.0, a)))
    return out


def mix(a: QColor, b: QColor, t: float) -> QColor:
    t = max(0.0, min(1.0, t))
    return QColor.fromRgbF(
        a.redF() + (b.redF() - a.redF()) * t,
        a.greenF() + (b.greenF() - a.greenF()) * t,
        a.blueF() + (b.blueF() - a.blueF()) * t,
        a.alphaF() + (b.alphaF() - a.alphaF()) * t,
    )


def _c(*args: float):
    return field(default_factory=lambda: rgba(*args))  # type: ignore[arg-type]


@dataclass(frozen=True)
class Theme:
    # surfaces
    glass: QColor = _c(12, 20, 28, 0.55)
    glass_strong: QColor = _c(16, 18, 21, 0.86)
    glass_highlight: QColor = _c(255, 255, 255, 0.06)
    surface: QColor = _c(255, 255, 255, 0.05)
    surface_hover: QColor = _c(255, 255, 255, 0.10)
    surface_active: QColor = _c(255, 255, 255, 0.16)
    border: QColor = _c(255, 255, 255, 0.13)
    border_strong: QColor = _c(255, 255, 255, 0.24)
    shadow: QColor = _c(0, 0, 0, 0.38)
    # type
    text: QColor = _c(244, 241, 234)
    text_secondary: QColor = _c(244, 241, 234, 0.62)
    text_tertiary: QColor = _c(244, 241, 234, 0.38)
    # accents
    accent: QColor = _c(156, 182, 214)  # cool blue-grey
    accent_soft: QColor = _c(156, 182, 214, 0.22)
    warm: QColor = _c(232, 196, 150)
    holo: QColor = _c(128, 222, 255)  # hologram rim / scanline tint
    success: QColor = _c(140, 199, 160)
    warning: QColor = _c(232, 190, 120)
    danger: QColor = _c(226, 122, 111)
    # metrics
    radius: float = 18.0
    radius_small: float = 10.0
    titlebar_height: float = 42.0
    control_height: float = 36.0
    scale: float = 1.0
    font_family: str = ""
    reduced_motion: bool = False
    high_contrast: bool = False

    def font(self, size: float, weight: QFont.Weight = QFont.Weight.Normal, letter_spacing: float = 0.0) -> QFont:
        f = QFont(self.font_family) if self.font_family else QFont(QApplication.font())
        f.setPointSizeF(size * self.scale)
        f.setWeight(weight)
        if letter_spacing:
            f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, letter_spacing)
        f.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
        return f

    def s(self, v: float) -> float:
        return v * self.scale


def make_theme(transparency: float = 0.55, window_scale: float = 1.0, large_buttons: bool = False,
               high_contrast: bool = False, reduced_motion: bool = False) -> Theme:
    base = Theme()
    scale = window_scale * (1.18 if large_buttons else 1.0)
    family = _preferred_family()
    t = replace(
        base,
        glass=rgba(12, 20, 28, transparency),
        glass_strong=rgba(16, 18, 21, min(0.95, transparency + 0.24)),
        scale=scale,
        titlebar_height=42.0 * scale,
        control_height=36.0 * scale,
        radius=18.0 * scale,
        radius_small=10.0 * scale,
        font_family=family,
        reduced_motion=reduced_motion,
        high_contrast=high_contrast,
    )
    if high_contrast:
        t = replace(
            t,
            glass=rgba(8, 9, 11, 0.92),
            glass_strong=rgba(4, 5, 6, 0.97),
            border=rgba(255, 255, 255, 0.55),
            border_strong=rgba(255, 255, 255, 0.85),
            text=rgba(255, 255, 255),
            text_secondary=rgba(255, 255, 255, 0.86),
            text_tertiary=rgba(255, 255, 255, 0.7),
            surface=rgba(255, 255, 255, 0.12),
            surface_hover=rgba(255, 255, 255, 0.22),
            accent=rgba(170, 205, 255),
        )
    return t


_FAMILY_CACHE: str | None = None


def _preferred_family() -> str:
    """Pick a clean, already-installed UI family; never bundles fonts."""
    global _FAMILY_CACHE
    if _FAMILY_CACHE is not None:
        return _FAMILY_CACHE
    try:
        families = set(QFontDatabase.families())
    except Exception:  # noqa: BLE001
        families = set()
    for name in ("SF Pro Text", "SF Pro Display", ".AppleSystemUIFont", "Helvetica Neue",
                 "Segoe UI Variable Text", "Segoe UI", "Inter", "Cantarell", "Ubuntu", "Noto Sans"):
        if name in families:
            _FAMILY_CACHE = name
            return name
    _FAMILY_CACHE = ""
    return ""
