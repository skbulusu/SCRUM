from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget

from scrum.app.overlay import animal_color
from scrum.app.project import Project
from scrum.app.theme import DARK, Theme
from scrum.app.timeline import GUTTER, RIGHT_PAD

PLOT_H = 46
GAP = 6


class TraceStrip(QWidget):
    seek_requested = pyqtSignal(int)

    def __init__(self, project: Project, theme: Theme = DARK):
        super().__init__()
        self.project = project
        self.theme = theme
        self.view_start, self.view_end = 0.0, 1.0
        self.frame = 0
        self._scales: list[float] = [1.0, 1.0]
        self.setMinimumHeight(PLOT_H * 2 + GAP * 3)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        project.data_changed.connect(self._on_data)
        self._on_data()

    def _on_data(self) -> None:
        p = self.project
        self.view_start, self.view_end = 0.0, float(max(1, p.n_frames))
        sp = [v for v in p.speed.values() if len(v)]
        sp_all = np.concatenate(sp) if sp else np.zeros(0)
        d = p.distance
        self._scales = [self._robust_max(sp_all), self._robust_max(d)]
        self.update()

    @staticmethod
    def _robust_max(a: np.ndarray) -> float:
        a = a[~np.isnan(a)] if len(a) else a
        return float(np.percentile(a, 99) * 1.1) if len(a) and np.percentile(a, 99) > 0 else 1.0

    def set_theme(self, theme: Theme) -> None:
        self.theme = theme
        self.update()

    def set_view(self, start: float, end: float) -> None:
        self.view_start, self.view_end = start, max(start + 1, end)
        self.update()

    def set_current_frame(self, frame: int) -> None:
        self.frame = frame
        self.update()


    def _plot_rects(self) -> list[QRectF]:
        w = max(10, self.width() - GUTTER - RIGHT_PAD)
        return [QRectF(GUTTER, GAP + i * (PLOT_H + GAP), w, PLOT_H) for i in range(2)]

    def _x(self, f: float, r: QRectF) -> float:
        return r.left() + (f - self.view_start) / (self.view_end - self.view_start) * r.width()

    def _path(self, data: np.ndarray, r: QRectF, ymax: float) -> QPainterPath:
        path = QPainterPath()
        n = len(data)
        if n == 0:
            return path
        width = int(r.width())
        span = self.view_end - self.view_start
        edges = np.clip((self.view_start + np.arange(width + 1) * span / width).astype(int), 0, n)
        starts, ends = edges[:-1], np.maximum(edges[1:], edges[:-1] + 1)
        ends = np.minimum(ends, n)
        starts = np.minimum(starts, n - 1)
        valid = ~np.isnan(data)
        filled = np.where(valid, data, 0.0)
        cs = np.concatenate([[0.0], np.cumsum(filled)])
        cv = np.concatenate([[0], np.cumsum(valid.astype(int))])
        counts = cv[ends] - cv[starts]
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(counts > 0, (cs[ends] - cs[starts]) / np.maximum(counts, 1), np.nan)
        pen_down = False
        for i, m in enumerate(means):
            if np.isnan(m):
                pen_down = False
                continue
            y = r.bottom() - 2 - min(1.0, m / ymax) * (r.height() - 4)
            pt = QPointF(r.left() + i + 0.5, y)
            if pen_down:
                path.lineTo(pt)
            else:
                path.moveTo(pt)
                pen_down = True
        return path


    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        t = self.theme
        proj = self.project
        rects = self._plot_rects()
        font = QFont(self.font())
        font.setPointSizeF(max(7.5, font.pointSizeF() - 1))
        p.setFont(font)
        titles = [f"Speed {proj.units}/s", f"Distance {proj.units}"]
        animals = proj.animals()
        for i, r in enumerate(rects):
            p.fillRect(r, QColor(t.track_bg))
            p.setPen(QPen(QColor(t.border), 1))
            p.drawRect(r)
            p.setPen(QColor(t.muted))
            p.drawText(QRectF(2, r.top(), GUTTER - 8, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, titles[i])
            p.drawText(QRectF(2, r.bottom() - 16, GUTTER - 8, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"0 to {self._scales[i]:.0f}")
        if proj.n_frames == 0 or not animals:
            p.setPen(QColor(t.muted))
            p.drawText(rects[0].united(rects[1]), Qt.AlignmentFlag.AlignCenter, "Traces appear once animals are tracked")
            return
        p.save()
        p.setClipRect(rects[0])
        for a in animals:
            p.setPen(QPen(self._qc(animal_color(a, animals)), 1.5))
            p.drawPath(self._path(proj.speed.get(a, np.zeros(0)), rects[0], self._scales[0]))
        p.restore()
        p.save()
        p.setClipRect(rects[1])
        p.setPen(QPen(QColor(t.text), 1.5))
        p.drawPath(self._path(proj.distance, rects[1], self._scales[1]))
        p.restore()

        x = rects[0].right() - 6
        for a in reversed(animals):
            p.setPen(self._qc(animal_color(a, animals)))
            w = p.fontMetrics().horizontalAdvance(a) + 14
            p.drawText(QPointF(x - w + 12, rects[0].top() + 12), a)
            p.fillRect(QRectF(x - w, rects[0].top() + 5, 8, 8), self._qc(animal_color(a, animals)))
            x -= w + 8

        for r in rects:
            xx = self._x(self.frame + 0.5, r)
            if r.left() <= xx <= r.right():
                p.setPen(QPen(QColor(t.playhead), 1.5))
                p.drawLine(QPointF(xx, r.top()), QPointF(xx, r.bottom()))
        self._paint_values(p, rects)

    def _paint_values(self, p: QPainter, rects: list[QRectF]) -> None:
        proj, f = self.project, self.frame
        if f >= proj.n_frames:
            return
        animals = proj.animals()
        speeds = " ".join(f"{a}: {proj.speed[a][f]:.1f}" for a in animals if a in proj.speed and not np.isnan(proj.speed[a][f]))
        dist = proj.distance[f] if len(proj.distance) > f else np.nan
        p.setPen(QColor(self.theme.muted))
        p.drawText(QRectF(rects[0].left() + 6, rects[0].top() + 1, rects[0].width() / 2, 14), Qt.AlignmentFlag.AlignLeft, speeds)
        if not np.isnan(dist):
            p.drawText(QRectF(rects[1].left() + 6, rects[1].top() + 1, 200, 14), Qt.AlignmentFlag.AlignLeft, f"{dist:.1f} {proj.units}")

    @staticmethod
    def _qc(bgr) -> QColor:
        return QColor(bgr[2], bgr[1], bgr[0])


    def mousePressEvent(self, ev) -> None:
        self._seek_to(ev.position().x())

    def mouseMoveEvent(self, ev) -> None:
        if ev.buttons() & Qt.MouseButton.LeftButton:
            self._seek_to(ev.position().x())

    def _seek_to(self, x: float) -> None:
        if self.project.n_frames == 0:
            return
        r = self._plot_rects()[0]
        f = self.view_start + (x - r.left()) / r.width() * (self.view_end - self.view_start)
        self.seek_requested.emit(int(max(0, min(self.project.n_frames - 1, round(f)))))
