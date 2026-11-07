from __future__ import annotations

import sys
from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

CHOICES = ("system", "light", "dark")


@dataclass(frozen=True)
class Theme:
    name: str
    window: str
    panel: str
    base: str
    text: str
    muted: str
    border: str
    accent: str
    video_bg: str
    track_bg: str
    grid: str
    playhead: str
    selection: str
    banner_info: str
    banner_warn: str
    banner_error: str


DARK = Theme(
    name="dark", window="#1e1e1e", panel="#2a2a2c", base="#242426", text="#ececec", muted="#98989d",
    border="#3d3d40", accent="#0a84ff", video_bg="#000000", track_bg="#2c2c2e", grid="#48484a",
    playhead="#ffffff", selection="#0a84ff", banner_info="#26313f", banner_warn="#3b3321", banner_error="#3f2624",
)
LIGHT = Theme(
    name="light", window="#ececec", panel="#f6f6f6", base="#ffffff", text="#1d1d1f", muted="#6e6e73",
    border="#d1d1d6", accent="#007aff", video_bg="#000000", track_bg="#e3e3e8", grid="#c7c7cc",
    playhead="#1d1d1f", selection="#007aff", banner_info="#dfeaf8", banner_warn="#f8efd6", banner_error="#f7dedc",
)
THEMES = {"dark": DARK, "light": LIGHT}


def is_mac() -> bool:
    return sys.platform == "darwin"


def _mix(a: QColor, b: QColor, t: float) -> str:
    c = QColor(
        round(a.red() + (b.red() - a.red()) * t),
        round(a.green() + (b.green() - a.green()) * t),
        round(a.blue() + (b.blue() - a.blue()) * t),
    )
    return c.name()


def theme_from_palette(pal: QPalette) -> Theme:
    window, base = pal.color(QPalette.ColorRole.Window), pal.color(QPalette.ColorRole.Base)
    text, accent = pal.color(QPalette.ColorRole.WindowText), pal.color(QPalette.ColorRole.Highlight)
    dark = window.lightness() < 128
    ink, paper = (QColor("#ffffff"), QColor("#000000")) if dark else (QColor("#000000"), QColor("#ffffff"))
    return Theme(
        name="dark" if dark else "light",
        window=window.name(),
        panel=_mix(window, ink, 0.04),
        base=base.name(),
        text=text.name(),
        muted=_mix(text, window, 0.42),
        border=_mix(window, ink, 0.16),
        accent=accent.name(),
        video_bg="#000000",
        track_bg=_mix(window, paper if dark else ink, 0.10 if dark else 0.07),
        grid=_mix(window, ink, 0.24),
        playhead=text.name(),
        selection=accent.name(),
        banner_info=_mix(window, QColor("#0a84ff"), 0.20),
        banner_warn=_mix(window, QColor("#ff9f0a"), 0.22),
        banner_error=_mix(window, QColor("#ff453a"), 0.22),
    )


def _supports_color_scheme() -> bool:
    return hasattr(Qt, "ColorScheme") and hasattr(QApplication.styleHints(), "setColorScheme")


def _fixed_palette(t: Theme) -> QPalette:
    pal = QPalette()
    for role, color in (
        (QPalette.ColorRole.Window, t.window), (QPalette.ColorRole.WindowText, t.text),
        (QPalette.ColorRole.Base, t.base), (QPalette.ColorRole.AlternateBase, t.panel),
        (QPalette.ColorRole.Text, t.text), (QPalette.ColorRole.Button, t.panel),
        (QPalette.ColorRole.ButtonText, t.text), (QPalette.ColorRole.Highlight, t.accent),
        (QPalette.ColorRole.HighlightedText, "#ffffff"), (QPalette.ColorRole.ToolTipBase, t.panel),
        (QPalette.ColorRole.ToolTipText, t.text), (QPalette.ColorRole.PlaceholderText, t.muted),
        (QPalette.ColorRole.Link, t.accent),
    ):
        pal.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText, QPalette.ColorRole.WindowText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor(t.muted))
    return pal


def apply_theme(app: QApplication, choice: str = "system") -> Theme:
    if choice not in CHOICES:
        choice = "system"
    hints = app.styleHints()
    app.setStyleSheet("")

    if _supports_color_scheme():
        forced = {"system": Qt.ColorScheme.Unknown, "light": Qt.ColorScheme.Light, "dark": Qt.ColorScheme.Dark}[choice]
        hints.setColorScheme(forced)

    if is_mac():
        if not app.style().objectName().lower().startswith("mac"):
            app.setStyle("macos")
        if _supports_color_scheme():
            app.setPalette(app.style().standardPalette())
            return theme_from_palette(app.palette())

        if choice == "system":
            app.setPalette(app.style().standardPalette())
            return theme_from_palette(app.palette())


    app.setStyle("Fusion")
    if choice == "system":
        dark = hasattr(Qt, "ColorScheme") and hasattr(hints, "colorScheme") and hints.colorScheme() == Qt.ColorScheme.Dark
        theme = DARK if dark else LIGHT
    else:
        theme = THEMES[choice]
    app.setPalette(_fixed_palette(theme))
    return theme
