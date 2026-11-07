from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QFont, QImage, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QMenu, QWidget

from scrum.app.analysis import BACKGROUND, runs_of_mask
from scrum.app.data import behavior_rgb
from scrum.app.project import Project
from scrum.app.theme import DARK, Theme

GUTTER = 78
RIGHT_PAD = 10
RULER_H = 20
ROW_H = 24
FLAG_H = 8
MINI_H = 26
GAP = 4
EDGE_PX = 5
MIN_SPAN = 20
_NICE_SECONDS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600]


def fmt_time(frame: float, fps: float) -> str:
    s = max(0.0, frame / (fps or 30.0))
    m, sec = divmod(s, 60)
    return f"{int(m)}:{sec:04.1f}"


class Timeline(QWidget):
    seek_requested = pyqtSignal(int)
    selection_changed = pyqtSignal(int, int)
    bout_selected = pyqtSignal(object)
    view_changed = pyqtSignal(float, float)
    status_message = pyqtSignal(str)

    def __init__(self, project: Project, theme: Theme = DARK):
        super().__init__()
        self.project = project
        self.theme = theme
        self.view_start = 0.0
        self.view_end = 1.0
        self.frame = 0
        self.selection: tuple[int, int] | None = None
        self.loop_range: tuple[int, int] | None = None
        self.selected_bout: dict | None = None
        self.review_items: list[dict] = []
        self.paint_mode = False
        self.paint_behavior = "sniffing"
        self.editable = True
        self.recording: tuple[int, int, str] | None = None

        self._drag: dict | None = None
        self._cache: dict[str, QImage] = {}
        self._cache_key = None
        self._row_keys: dict[str, tuple] = {}
        self._mini_key = None
        self._ids_version = 0

        self.setMouseTracking(True)
        self.setMinimumHeight(self._total_height())
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        project.labels_changed.connect(self._invalidate)
        project.data_changed.connect(self._on_data_changed)
        project.bookmarks_changed.connect(self.update)
        self._on_data_changed()


    def _total_height(self) -> int:
        return RULER_H + GAP + ROW_H * 2 + GAP + FLAG_H + GAP + MINI_H + GAP

    def _rows(self) -> dict[str, QRectF]:
        w = self.width()
        x, rw = GUTTER, max(10, w - GUTTER - RIGHT_PAD)
        y = 0.0
        out = {}
        for name, h in (("ruler", RULER_H), ("model", ROW_H), ("final", ROW_H), ("flags", FLAG_H), ("mini", MINI_H)):
            out[name] = QRectF(x, y, rw, h)
            y += h + GAP
        return out

    @property
    def n(self) -> int:
        return max(1, self.project.n_frames)

    def _span(self) -> float:
        return max(1.0, self.view_end - self.view_start)

    def frame_to_x(self, f: float) -> float:
        r = self._rows()["ruler"]
        return r.left() + (f - self.view_start) / self._span() * r.width()

    def x_to_frame(self, x: float) -> float:
        r = self._rows()["ruler"]
        return self.view_start + (x - r.left()) / r.width() * self._span()

    def _mini_x_to_frame(self, x: float) -> float:
        r = self._rows()["mini"]
        return (x - r.left()) / r.width() * self.n

    def _clamp_frame(self, f: float) -> int:
        return int(max(0, min(self.n - 1, round(f))))


    def _on_data_changed(self) -> None:
        self.view_start, self.view_end = 0.0, float(self.n)
        self.selection = None
        self.selected_bout = None
        self.loop_range = None
        self._invalidate()
        self.view_changed.emit(self.view_start, self.view_end)

    def _invalidate(self) -> None:
        self._cache.clear()
        self._cache_key = None
        self._ids_version += 1
        self.update()

    def set_theme(self, theme: Theme) -> None:
        self.theme = theme
        self._invalidate()

    def set_view(self, start: float, end: float) -> None:
        span = max(MIN_SPAN, end - start)
        span = min(span, float(self.n))
        start = max(0.0, min(start, self.n - span))
        self.view_start, self.view_end = start, start + span
        self._cache.clear()
        self.view_changed.emit(self.view_start, self.view_end)
        self.update()

    def zoom(self, factor: float, anchor_frame: float | None = None) -> None:
        anchor = self.frame if anchor_frame is None else anchor_frame
        span = self._span() / factor
        frac = (anchor - self.view_start) / self._span()
        start = anchor - frac * span
        self.set_view(start, start + span)

    def zoom_in(self) -> None:
        self.zoom(1.6)

    def zoom_out(self) -> None:
        self.zoom(1 / 1.6)

    def fit(self) -> None:
        self.set_view(0, self.n)

    def ensure_visible(self, frame: int) -> None:
        if frame < self.view_start or frame > self.view_end:
            span = self._span()
            self.set_view(frame - span * 0.15, frame - span * 0.15 + span)

    def set_current_frame(self, frame: int, follow: bool = True) -> None:
        self.frame = int(frame)
        if follow and not self._drag:
            self.ensure_visible(self.frame)
        self.update()

    def set_selection(self, sel: tuple[int, int] | None) -> None:
        self.selection = (min(sel), max(sel)) if sel else None
        if self.selection:
            self.selection_changed.emit(*self.selection)
        else:
            self.selection_changed.emit(-1, -1)
        self.update()

    def set_loop_range(self, rng: tuple[int, int] | None) -> None:
        self.loop_range = rng
        self.update()

    def set_selected_bout(self, bout: dict | None) -> None:
        self.selected_bout = bout
        self.bout_selected.emit(bout)
        self.update()

    def set_recording(self, rec: tuple[int, int, str] | None) -> None:
        self.recording = rec
        self.update()

    def set_review_items(self, items: list[dict]) -> None:
        self.review_items = items
        self._cache.pop("flags", None)
        self.update()


    def _label_ids(self, labels: np.ndarray, vocab: dict[str, int]) -> np.ndarray:
        ids = np.zeros(len(labels), dtype=np.int16)
        for name, i in vocab.items():
            if name not in ("", BACKGROUND):
                ids[labels == name] = i
        return ids

    def _row_image(self, name: str, width: int) -> QImage:
        p = self.project
        key = (name, width, round(self.view_start, 2), round(self.view_end, 2), self._ids_version)
        cached = self._cache.get(name)
        if cached is not None and self._row_keys.get(name) == key:
            return cached
        t = self.theme
        bg = QColor(t.track_bg)
        img = np.zeros((ROW_H, width, 4), dtype=np.uint8)
        img[..., :3] = (bg.red(), bg.green(), bg.blue())
        img[..., 3] = 255
        n = len(p.pred)
        if n:
            labels = p.pred if name == "model" else p.final_labels()
            labels = np.where(labels == "", BACKGROUND, labels)
            vocab = {b: i for i, b in enumerate(p.behaviors())}
            ids = self._label_ids(labels, vocab)
            starts = np.floor(self.view_start + np.arange(width) * self._span() / width).astype(int)
            starts = np.clip(starts, 0, n - 1)
            col_ids = np.maximum.reduceat(ids, starts)
            if name == "model" and not np.all(np.isnan(p.conf)):
                conf = np.where(np.isnan(p.conf), 1.0, p.conf)
                cs = np.concatenate([[0.0], np.cumsum(conf)])
                ends = np.minimum(np.append(starts[1:], int(np.ceil(self.view_end))) + 0, n)
                ends = np.maximum(ends, starts + 1)
                col_conf = (cs[ends] - cs[starts]) / (ends - starts)
            else:
                col_conf = np.ones(width)
            inv = {i: b for b, i in vocab.items()}
            bgc = np.array([bg.red(), bg.green(), bg.blue()], dtype=float)
            for i in np.unique(col_ids):
                if i == 0:
                    continue
                rgb = np.array(behavior_rgb(inv[int(i)]), dtype=float)
                cols = np.nonzero(col_ids == i)[0]
                shade = (0.30 + 0.70 * np.clip(col_conf[cols], 0, 1))[:, None]
                img[:, cols, :3] = (rgb * shade + bgc * (1 - shade)).astype(np.uint8)[None, :, :]
        qimg = QImage(img.data, width, ROW_H, width * 4, QImage.Format.Format_RGBA8888).copy()
        self._cache[name] = qimg
        self._row_keys[name] = key
        return qimg

    def _mini_image(self, width: int) -> QImage:
        key = (width, self._ids_version, self.theme.name)
        if self._cache.get("mini") is not None and self._mini_key == key:
            return self._cache["mini"]
        p = self.project
        bg = QColor(self.theme.track_bg)
        img = np.zeros((MINI_H, width, 4), dtype=np.uint8)
        img[..., :3] = (bg.red(), bg.green(), bg.blue())
        img[..., 3] = 255
        n = len(p.pred)
        if n:
            labels = np.where(p.final_labels() == "", BACKGROUND, p.final_labels())
            vocab = {b: i for i, b in enumerate(p.behaviors())}
            ids = self._label_ids(labels, vocab)
            starts = np.clip((np.arange(width) * n / width).astype(int), 0, n - 1)
            col_ids = np.maximum.reduceat(ids, starts)
            inv = {i: b for b, i in vocab.items()}
            for i in np.unique(col_ids):
                if i == 0:
                    continue
                img[:, col_ids == i, :3] = behavior_rgb(inv[int(i)])
        qimg = QImage(img.data, width, MINI_H, width * 4, QImage.Format.Format_RGBA8888).copy()
        self._cache["mini"] = qimg
        self._mini_key = key
        return qimg


    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        t = self.theme
        rows = self._rows()
        proj = self.project
        font = QFont(self.font())
        font.setPointSizeF(max(7.5, font.pointSizeF() - 1))
        p.setFont(font)
        muted, text = QColor(t.muted), QColor(t.text)

        for label, key in (("Model", "model"), ("Final", "final"), ("Flags", "flags"), ("Overview", "mini")):
            r = rows[key]
            p.setPen(muted)
            p.drawText(QRectF(4, r.top(), GUTTER - 8, r.height()), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, label)

        w = int(rows["ruler"].width())
        if proj.n_frames == 0:
            p.setPen(muted)
            p.drawText(rows["model"].united(rows["final"]), Qt.AlignmentFlag.AlignCenter, "Open a video to see the timeline")
            return

        self._paint_ruler(p, rows["ruler"], muted)

        for key in ("model", "final"):
            r = rows[key]
            p.drawImage(QPointF(r.left(), r.top()), self._row_image(key, w))
            p.setPen(QPen(QColor(t.border), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(r)

        self._paint_flags(p, rows["flags"], w)
        self._paint_bout_highlights(p, rows)
        self._paint_overlays(p, rows)
        self._paint_minimap(p, rows["mini"], w)
        self._paint_playhead(p, rows)

    def _paint_ruler(self, p: QPainter, r: QRectF, muted: QColor) -> None:
        fps = self.project.fps or 30.0
        span_s = self._span() / fps
        pxs = r.width() / max(span_s, 1e-6)
        step = next((s for s in _NICE_SECONDS if s * pxs >= 80), _NICE_SECONDS[-1])
        t0 = int(self.view_start / fps / step) * step
        p.setPen(QPen(QColor(self.theme.grid), 1))
        k = t0
        while k * fps <= self.view_end + 1:
            x = self.frame_to_x(k * fps)
            if r.left() - 1 <= x <= r.right() + 1:
                p.drawLine(QPointF(x, r.bottom() - 6), QPointF(x, r.bottom()))
                p.setPen(muted)
                label = f"{int(k // 60)}:{k % 60:04.1f}" if step < 1 else f"{int(k // 60)}:{int(k % 60):02d}"
                p.drawText(QPointF(x + 3, r.bottom() - 7), label)
                p.setPen(QPen(QColor(self.theme.grid), 1))
            k = round(k + step, 6)

        for b in self.project.bookmarks:
            x = self.frame_to_x(b["frame"])
            if r.left() <= x <= r.right():
                p.setBrush(QColor("#f4b400"))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawPolygon(QPolygonF([QPointF(x, r.top() + 1), QPointF(x + 7, r.top() + 5), QPointF(x, r.top() + 9)]))
                p.setPen(QPen(QColor("#f4b400"), 1))
                p.drawLine(QPointF(x, r.top() + 1), QPointF(x, r.bottom()))
        p.setBrush(Qt.BrushStyle.NoBrush)

        if self.loop_range:
            a, b = (self.frame_to_x(v) for v in (self.loop_range[0], self.loop_range[1] + 1))
            p.setPen(QPen(QColor(self.theme.accent), 3))
            p.drawLine(QPointF(max(a, r.left()), r.bottom() - 1), QPointF(min(b, r.right()), r.bottom() - 1))

    def _paint_flags(self, p: QPainter, r: QRectF, w: int) -> None:
        proj = self.project
        p.fillRect(r, QColor(self.theme.track_bg))
        if len(proj.merged):
            for s, e in runs_of_mask(proj.merged):
                a, b = self.frame_to_x(s), self.frame_to_x(e + 1)
                if b >= r.left() and a <= r.right():
                    p.fillRect(QRectF(max(a, r.left()), r.top(), max(1.0, min(b, r.right()) - max(a, r.left())), r.height() / 2), QColor("#f4a300"))
        for it in self.review_items:
            if it["kind"] != "low confidence":
                continue
            a, b = self.frame_to_x(it["start"]), self.frame_to_x(it["end"] + 1)
            if b >= r.left() and a <= r.right():
                p.fillRect(QRectF(max(a, r.left()), r.top() + r.height() / 2, max(1.0, min(b, r.right()) - max(a, r.left())), r.height() / 2), QColor("#e5484d"))
        p.setPen(QPen(QColor(self.theme.border), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(r)

    def _paint_bout_highlights(self, p: QPainter, rows: dict[str, QRectF]) -> None:
        r = rows["final"]
        if self.selected_bout:
            b = self.selected_bout
            a, c = self.frame_to_x(b["start"]), self.frame_to_x(b["end"] + 1)
            p.setPen(QPen(QColor("#ffffff"), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(a, r.top() + 1, max(2.0, c - a), r.height() - 2))
        d = self._drag
        if d and d.get("kind") == "resize":
            a, c = self.frame_to_x(d["new_start"]), self.frame_to_x(d["new_end"] + 1)
            col = QColor(*behavior_rgb(d["bout"]["behavior"]))
            col.setAlpha(150)
            p.fillRect(QRectF(a, r.top(), max(2.0, c - a), r.height()), col)
            p.setPen(QPen(QColor("#ffffff"), 2, Qt.PenStyle.DashLine))
            p.drawRect(QRectF(a, r.top(), max(2.0, c - a), r.height()))
        if self.recording:
            lo, hi, beh = self.recording
            a, c = self.frame_to_x(lo), self.frame_to_x(hi + 1)
            col = QColor(*behavior_rgb(beh))
            col.setAlpha(200)
            p.fillRect(QRectF(a, r.top(), max(2.0, c - a), r.height()), col)
            p.setPen(QPen(QColor("#ff3b30"), 2))
            p.drawRect(QRectF(a, r.top(), max(2.0, c - a), r.height()))
        if d and d.get("kind") == "paint":
            lo, hi = sorted((d["f0"], d["f1"]))
            a, c = self.frame_to_x(lo), self.frame_to_x(hi + 1)
            col = QColor(*behavior_rgb(self.paint_behavior))
            col.setAlpha(170)
            p.fillRect(QRectF(a, r.top(), max(2.0, c - a), r.height()), col)

    def _paint_overlays(self, p: QPainter, rows: dict[str, QRectF]) -> None:
        top, bottom = rows["ruler"].bottom(), rows["flags"].bottom()
        if self.selection:
            a, b = self.frame_to_x(self.selection[0]), self.frame_to_x(self.selection[1] + 1)
            col = QColor(self.theme.selection)
            col.setAlpha(70)
            p.fillRect(QRectF(a, top, max(2.0, b - a), bottom - top), col)
            p.setPen(QPen(QColor(self.theme.selection), 1))
            p.drawLine(QPointF(a, top), QPointF(a, bottom))
            p.drawLine(QPointF(b, top), QPointF(b, bottom))

    def _paint_minimap(self, p: QPainter, r: QRectF, w: int) -> None:
        p.drawImage(QPointF(r.left(), r.top()), self._mini_image(w))
        p.setPen(QPen(QColor(self.theme.border), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(r)
        n = self.n
        a = r.left() + self.view_start / n * r.width()
        b = r.left() + self.view_end / n * r.width()
        shade = QColor(0, 0, 0, 90 if self.theme.name == "dark" else 50)
        p.fillRect(QRectF(r.left(), r.top(), a - r.left(), r.height()), shade)
        p.fillRect(QRectF(b, r.top(), r.right() - b, r.height()), shade)
        p.setPen(QPen(QColor(self.theme.accent), 2))
        p.drawRect(QRectF(a, r.top() + 1, max(3.0, b - a), r.height() - 2))
        x = r.left() + self.frame / n * r.width()
        p.setPen(QPen(QColor(self.theme.playhead), 1))
        p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))

    def _paint_playhead(self, p: QPainter, rows: dict[str, QRectF]) -> None:
        x = self.frame_to_x(self.frame + 0.5)
        r0, r1 = rows["ruler"], rows["flags"]
        if r0.left() <= x <= r0.right():
            p.setPen(QPen(QColor(self.theme.playhead), 2))
            p.drawLine(QPointF(x, r0.top()), QPointF(x, r1.bottom()))
            p.setBrush(QColor(self.theme.playhead))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPolygon(QPolygonF([QPointF(x - 5, r0.top()), QPointF(x + 5, r0.top()), QPointF(x, r0.top() + 7)]))


    def _row_at(self, pos) -> str | None:
        for name, r in self._rows().items():
            if r.adjusted(0, -GAP / 2, 0, GAP / 2).contains(pos):
                return name
        return None

    def _edge_hit(self, pos) -> tuple[dict, str] | None:
        if not self.editable or self._row_at(pos) != "final":
            return None
        f = self.x_to_frame(pos.x())
        for b in self.project.bouts():
            if b["end"] + 1 < f - 40 or b["start"] > f + 40:
                continue
            if abs(pos.x() - self.frame_to_x(b["start"])) <= EDGE_PX:
                return b, "start"
            if abs(pos.x() - self.frame_to_x(b["end"] + 1)) <= EDGE_PX:
                return b, "end"
        return None

    def mousePressEvent(self, ev) -> None:
        pos = ev.position()
        row = self._row_at(pos)
        if self.project.n_frames == 0 or row is None:
            return
        mods = ev.modifiers()
        if ev.button() == Qt.MouseButton.RightButton:
            self._context_menu(ev.globalPosition().toPoint(), pos)
            return
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        if row == "mini":
            self._drag = {"kind": "mini", "offset": self._mini_x_to_frame(pos.x()) - (self.view_start + self.view_end) / 2}
            self._move_mini(pos.x())
            return
        f = self._clamp_frame(self.x_to_frame(pos.x()))
        if row == "ruler":
            self.set_selection(None)
            bm = next((b for b in self.project.bookmarks if abs(self.frame_to_x(b["frame"]) - pos.x()) <= 6), None)
            if bm:
                self.seek_requested.emit(bm["frame"])
                return
        if self.editable and (self.paint_mode or mods & Qt.KeyboardModifier.AltModifier) and row in ("final", "model"):
            self._drag = {"kind": "paint", "f0": f, "f1": f}
        elif mods & Qt.KeyboardModifier.ShiftModifier and row in ("model", "final"):
            self._drag = {"kind": "select", "f0": f}
            self.set_selection((f, f))
        else:
            edge = self._edge_hit(pos)
            if edge:
                bout, which = edge
                self._drag = {"kind": "resize", "bout": bout, "which": which, "new_start": bout["start"], "new_end": bout["end"]}
                return
            self._drag = {"kind": "scrub"}
            if row == "final":
                bout = self.project.bout_at(f)
                self.set_selected_bout(bout)
            self.seek_requested.emit(f)
        self.update()

    def mouseMoveEvent(self, ev) -> None:
        pos = ev.position()
        d = self._drag
        if d is None:
            edge = self._edge_hit(pos)
            self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor if edge else Qt.CursorShape.ArrowCursor))
            if self.project.n_frames and self._row_at(pos) in ("model", "final", "flags"):
                f = self._clamp_frame(self.x_to_frame(pos.x()))
                self.setToolTip(self._tooltip(f))
            return
        kind = d["kind"]
        if kind == "mini":
            self._move_mini(pos.x())
            return
        f = self._clamp_frame(self.x_to_frame(pos.x()))
        if kind == "scrub":
            self.seek_requested.emit(f)
        elif kind == "select":
            self.set_selection((d["f0"], f))
        elif kind == "paint":
            d["f1"] = f
        elif kind == "resize":
            b = d["bout"]
            if d["which"] == "start":
                d["new_start"] = min(f, b["end"])
            else:
                d["new_end"] = max(f - 1, b["start"]) if f > b["start"] else b["start"]
        self.update()

    def mouseReleaseEvent(self, ev) -> None:
        d, self._drag = self._drag, None
        if not d:
            return
        if d["kind"] == "resize":
            b = d["bout"]
            if (d["new_start"], d["new_end"]) != (b["start"], b["end"]):
                if self.project.resize_bout(b, d["new_start"], d["new_end"]):
                    self.set_selected_bout(self.project.bout_at(d["new_start"]))
        elif d["kind"] == "paint":
            lo, hi = sorted((d["f0"], d["f1"]))
            if self.project.add_bout(lo, hi, self.paint_behavior):
                self.set_selected_bout(self.project.bout_at(lo))
        self.update()

    def mouseDoubleClickEvent(self, ev) -> None:
        pos = ev.position()
        if self._row_at(pos) in ("model", "final"):
            f = self._clamp_frame(self.x_to_frame(pos.x()))
            bout = self.project.bout_at(f)
            if bout:
                self.set_selected_bout(bout)
                self.set_selection((bout["start"], bout["end"]))

    def wheelEvent(self, ev) -> None:
        if self.project.n_frames == 0:
            return
        delta = ev.angleDelta().y() or ev.angleDelta().x()
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            anchor = self.x_to_frame(ev.position().x())
            self.zoom(1.25 if delta > 0 else 1 / 1.25, anchor)
        else:
            shift = -delta / 120 * self._span() * 0.12
            self.set_view(self.view_start + shift, self.view_end + shift)
        ev.accept()

    def _move_mini(self, x: float) -> None:
        center = self._mini_x_to_frame(x) - (self._drag or {}).get("offset", 0.0)
        half = self._span() / 2
        self.set_view(center - half, center + half)

    def _tooltip(self, f: int) -> str:
        p = self.project
        parts = [f"frame {f}  {fmt_time(f, p.fps)}"]
        if f < len(p.pred):
            model = p.pred[f] or "none"
            parts.append(f"model: {model}" + (f" ({p.conf[f]:.2f})" if not np.isnan(p.conf[f]) else ""))
            parts.append(f"final: {p.label_at(f)}" + (" (edited)" if p.manual[f] != "" else ""))
            if len(p.merged) > f and p.merged[f]:
                parts.append("animals merged into one blob")
        return "\n".join(parts)


    def _context_menu(self, global_pos, pos) -> None:
        proj = self.project
        f = self._clamp_frame(self.x_to_frame(pos.x()))
        bout = proj.bout_at(f)
        menu = QMenu(self)
        menu.addAction(f"Seek to frame {f}", lambda: self.seek_requested.emit(f))
        bm = proj.bookmark_near(f, tolerance=max(1, int(self._span() / max(1, self.width()) * 6)))
        if bm:
            menu.addAction("Remove bookmark", lambda: proj.remove_bookmark(bm["frame"]))
        else:
            menu.addAction("Add bookmark here", lambda: proj.add_bookmark(f, ""))
        if self.editable:
            menu.addSeparator()
            if bout:
                menu.addAction(f"Split {bout['behavior']} bout here", lambda: proj.split_bout(f))
                nxt = next((b for b in proj.bouts() if b["start"] > bout["end"]), None)
                if nxt:
                    menu.addAction(f"Merge with next bout ({nxt['behavior']})", lambda: proj.merge_with_next(bout))
                sub = menu.addMenu("Relabel bout as")
                for beh in proj.behaviors():
                    sub.addAction(beh, lambda b=beh: proj.relabel_bout(bout, b))
                menu.addAction("Delete bout", lambda: proj.delete_bout(bout))
            if self.selection:
                sub = menu.addMenu("Label selection as")
                for beh in proj.behaviors():
                    sub.addAction(beh, lambda b=beh: (proj.set_range(self.selection[0], self.selection[1], b), self.set_selection(None)))
            if proj.can_undo():
                menu.addAction("Undo", proj.undo)
        if self.selection:
            menu.addSeparator()
            menu.addAction("Clear selection", lambda: self.set_selection(None))
        menu.exec(global_pos)

    def resizeEvent(self, _e) -> None:
        self._cache.clear()
        self._cache_key = None

    def sizeHint(self):
        from PyQt6.QtCore import QSize

        return QSize(900, self._total_height())
