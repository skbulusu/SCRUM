from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

from scrum.app.theme import DARK, Theme


class VideoView(QWidget):
    point_clicked = pyqtSignal(float, float)
    point_double_clicked = pyqtSignal(float, float)

    def __init__(self, theme: Theme = DARK):
        super().__init__()
        self.theme = theme
        self._image: QImage | None = None
        self._message = "Open a video to begin (File > Open Video, or drop one here)"
        self._hint = ""
        self._pending: list[tuple[float, float]] = []
        self._pending_closed = False
        self.setMinimumSize(480, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self._pick_mode = False


    def set_image(self, image: QImage | None) -> None:
        self._image = image
        self.update()

    def set_message(self, text: str) -> None:
        self._message = text
        self.update()

    def set_hint(self, text: str) -> None:
        self._hint = text
        self.update()

    def set_theme(self, theme: Theme) -> None:
        self.theme = theme
        self.update()

    def set_pick_mode(self, on: bool) -> None:
        self._pick_mode = on
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor))
        if not on:
            self._pending, self._pending_closed, self._hint = [], False, ""
        self.update()

    def set_pending(self, points: list[tuple[float, float]], closed: bool = False) -> None:
        self._pending, self._pending_closed = list(points), closed
        self.update()


    def image_rect(self) -> QRectF | None:
        if self._image is None or self._image.isNull():
            return None
        iw, ih = self._image.width(), self._image.height()
        scale = min(self.width() / iw, self.height() / ih)
        w, h = iw * scale, ih * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def to_frame(self, pos: QPointF) -> tuple[float, float] | None:
        r = self.image_rect()
        if r is None or not r.contains(pos):
            return None
        return ((pos.x() - r.left()) / r.width() * self._image.width(), (pos.y() - r.top()) / r.height() * self._image.height())

    def to_widget(self, x: float, y: float) -> QPointF:
        r = self.image_rect()
        if r is None:
            return QPointF(x, y)
        return QPointF(r.left() + x / self._image.width() * r.width(), r.top() + y / self._image.height() * r.height())


    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(self.theme.video_bg))
        r = self.image_rect()
        if r is not None:
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            p.drawImage(r, self._image)
            self._paint_pending(p)
            if self._hint:
                self._paint_banner(p, self._hint, QColor(self.theme.accent), top=True)
        else:
            p.setPen(QColor(self.theme.muted))
            f = QFont(self.font())
            f.setPointSizeF(f.pointSizeF() + 2)
            p.setFont(f)
            p.drawText(self.rect().adjusted(30, 30, -30, -30), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self._message)

    def _paint_pending(self, p: QPainter) -> None:
        if not self._pending:
            return
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pts = [self.to_widget(x, y) for x, y in self._pending]
        p.setPen(QPen(QColor("#ff5bd8"), 2))
        for a, b in zip(pts, pts[1:]):
            p.drawLine(a, b)
        if self._pending_closed and len(pts) > 2:
            p.drawLine(pts[-1], pts[0])
        p.setBrush(QColor("#ff5bd8"))
        for i, pt in enumerate(pts):
            p.drawEllipse(pt, 5, 5)
            p.setPen(QColor("#ffffff"))
            p.drawText(pt + QPointF(8, -8), str(i + 1))
            p.setPen(QPen(QColor("#ff5bd8"), 2))

    def _paint_banner(self, p: QPainter, text: str, color: QColor, top: bool = True) -> None:
        r = self.image_rect()
        if r is None:
            return
        f = QFont(self.font())
        f.setBold(True)
        p.setFont(f)
        box = QRectF(r.left() + 10, r.top() + 10, min(r.width() - 20, 640), 30)
        p.setPen(Qt.PenStyle.NoPen)
        c = QColor(color)
        c.setAlpha(225)
        p.setBrush(c)
        p.drawRoundedRect(box, 5, 5)
        p.setPen(QColor("#ffffff"))
        p.drawText(box.adjusted(10, 0, -10, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)


    def mousePressEvent(self, ev) -> None:
        if self._pick_mode and ev.button() == Qt.MouseButton.LeftButton:
            pt = self.to_frame(ev.position())
            if pt:
                self.point_clicked.emit(*pt)

    def mouseDoubleClickEvent(self, ev) -> None:
        if self._pick_mode and ev.button() == Qt.MouseButton.LeftButton:
            pt = self.to_frame(ev.position())
            if pt:
                self.point_double_clicked.emit(*pt)
