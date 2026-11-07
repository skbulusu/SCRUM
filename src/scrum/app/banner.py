from __future__ import annotations

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from scrum.app.theme import DARK, Theme


class StatusBanner(QWidget):
    def __init__(self, theme: Theme = DARK):
        super().__init__()
        self.theme = theme
        self._level = "info"
        self._action_cb = None
        self.label = QLabel("")
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.action = QPushButton("")
        self.action.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.action.clicked.connect(self._on_action)
        self.close_btn = QPushButton("Dismiss")
        self.close_btn.setFlat(True)
        self.close_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.close_btn.clicked.connect(self.hide)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 5, 6, 5)
        lay.addWidget(self.label, 1)
        lay.addWidget(self.action)
        lay.addWidget(self.close_btn)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)
        self.hide()

    def set_theme(self, theme: Theme) -> None:
        self.theme = theme
        self._style()

    def _style(self) -> None:
        bg = {"info": self.theme.banner_info, "warn": self.theme.banner_warn, "error": self.theme.banner_error}[self._level]
        self.setStyleSheet(f"StatusBanner {{ background-color: {bg}; border-radius: 6px; }} QLabel {{ background: transparent; }}")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    def show_message(self, text: str, level: str = "info", action_text: str = "", action=None, timeout_ms: int = 0) -> None:
        self._level = level
        self._action_cb = action
        self.label.setText(text)
        self.action.setText(action_text)
        self.action.setVisible(bool(action_text))
        self._style()
        self.show()
        self._timer.stop()
        if timeout_ms > 0:
            self._timer.start(timeout_ms)

    def _on_action(self) -> None:
        if self._action_cb:
            self._action_cb()

    @property
    def text(self) -> str:
        return self.label.text() if self.isVisible() else ""
