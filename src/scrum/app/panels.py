from __future__ import annotations

import math

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from scrum.app import analysis, shortcuts
from scrum.app.analysis import BACKGROUND
from scrum.app.data import behavior_rgb
from scrum.app.project import Project
from scrum.app.timeline import fmt_time


def _chip(name: str) -> QPixmap:
    pm = QPixmap(14, 14)
    r, g, b = behavior_rgb(name)
    pm.fill(QColor(r, g, b))
    return pm


def _item(text: str, align_right: bool = False) -> QTableWidgetItem:
    it = QTableWidgetItem(text)
    it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
    if align_right:
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return it


class SummaryPanel(QWidget):
    def __init__(self, project: Project):
        super().__init__()
        self.project = project
        self.info = QLabel("")
        self.info.setObjectName("muted")
        self.info.setWordWrap(True)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Behavior", "Time (s)", "% video", "Bouts", "Mean bout (s)", "Latency (s)"])
        self.table.horizontalHeaderItem(5).setToolTip("Time from video start to first bout")
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(self.table)
        lay.addWidget(self.info)
        project.labels_changed.connect(self.refresh)
        project.data_changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        p = self.project
        if not p.has_video:
            self.table.setRowCount(0)
            self.info.setText("")
            return
        s = p.summary()
        self.table.setRowCount(len(s))
        for i, r in enumerate(s.itertuples()):
            name_item = _item(r.behavior)
            name_item.setIcon(_icon(r.behavior))
            self.table.setItem(i, 0, name_item)
            self.table.setItem(i, 1, _item(f"{r.seconds:.1f}", True))
            self.table.setItem(i, 2, _item(f"{r.percent:.1f}", True))
            self.table.setItem(i, 3, _item(str(int(r.bouts)), True))
            self.table.setItem(i, 4, _item(f"{r.mean_bout_s:.2f}", True))
            self.table.setItem(i, 5, _item("never" if math.isnan(r.latency_s) else f"{r.latency_s:.1f}", True))
        merged = 100.0 * p.merged.mean() if len(p.merged) else 0.0
        edited = p.n_manual_frames()
        parts = [f"{p.n_frames} frames", f"{p.n_frames / (p.fps or 30):.1f} s", f"{merged:.0f}% fused"]
        if edited:
            parts.append(f"{edited} edited")
        self.info.setText("  ·  ".join(parts))


def _icon(name: str):
    from PyQt6.QtGui import QIcon

    return QIcon(_chip(name))


class BookmarksPanel(QWidget):
    seek_requested = pyqtSignal(int)

    def __init__(self, project: Project, current_frame):
        super().__init__()
        self.project = project
        self._current_frame = current_frame
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(self._open)
        add = QPushButton("Add at playhead")
        add.clicked.connect(self._add)
        edit = QPushButton("Edit note")
        edit.clicked.connect(self._edit)
        rem = QPushButton("Remove")
        rem.clicked.connect(self._remove)
        row = QHBoxLayout()
        for b in (add, edit, rem):
            row.addWidget(b)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(QLabel("Double-click to jump. Notes are saved in the session."))
        lay.addWidget(self.list)
        lay.addLayout(row)
        project.bookmarks_changed.connect(self.refresh)
        project.data_changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        self.list.clear()
        for b in self.project.bookmarks:
            note = f"  {b['note']}" if b["note"] else ""
            it = QListWidgetItem(f"frame {b['frame']}  ({fmt_time(b['frame'], self.project.fps)}){note}")
            it.setData(Qt.ItemDataRole.UserRole, b["frame"])
            self.list.addItem(it)

    def _selected_frame(self) -> int | None:
        it = self.list.currentItem()
        return it.data(Qt.ItemDataRole.UserRole) if it else None

    def _open(self, it) -> None:
        self.seek_requested.emit(it.data(Qt.ItemDataRole.UserRole))

    def _add(self) -> None:
        if not self.project.has_video:
            return
        f = self._current_frame()
        existing = self.project.bookmark_near(f)
        note, ok = QInputDialog.getText(self, "Bookmark", f"Note for frame {f}:", text=existing["note"] if existing else "")
        if ok:
            self.project.add_bookmark(f, note)

    def _edit(self) -> None:
        f = self._selected_frame()
        if f is None:
            return
        b = self.project.bookmark_near(f)
        note, ok = QInputDialog.getText(self, "Edit note", f"Note for frame {f}:", text=b["note"] if b else "")
        if ok:
            self.project.add_bookmark(f, note)

    def _remove(self) -> None:
        f = self._selected_frame()
        if f is not None:
            self.project.remove_bookmark(f)


class ReviewPanel(QWidget):
    jump_requested = pyqtSignal(int, int)
    items_changed = pyqtSignal(object)

    def __init__(self, project: Project):
        super().__init__()
        self.project = project
        self.items: list[dict] = []
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.2, 0.99)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(0.6)
        self.min_frames = QSpinBox()
        self.min_frames.setRange(1, 300)
        self.min_frames.setValue(5)
        self.include_merged = QCheckBox("Include merged-blob stretches")
        self.include_merged.setChecked(True)
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(lambda it: self._jump(self.list.row(it)))
        prev_b, next_b = QPushButton("Previous"), QPushButton("Next")
        prev_b.clicked.connect(lambda: self.step(-1))
        next_b.clicked.connect(lambda: self.step(1))
        form = QHBoxLayout()
        form.addWidget(QLabel("Confidence below"))
        form.addWidget(self.threshold)
        form.addWidget(QLabel("for at least"))
        form.addWidget(self.min_frames)
        form.addWidget(QLabel("frames"))
        form.addStretch(1)
        nav = QHBoxLayout()
        nav.addWidget(prev_b)
        nav.addWidget(next_b)
        self.status = QLabel("")
        self.status.setObjectName("muted")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addLayout(form)
        lay.addWidget(self.include_merged)
        lay.addWidget(self.list)
        lay.addLayout(nav)
        lay.addWidget(self.status)
        for w in (self.threshold, self.min_frames):
            w.valueChanged.connect(self.refresh)
        self.include_merged.toggled.connect(self.refresh)
        project.data_changed.connect(self.refresh)
        project.labels_changed.connect(self._mark_edited)
        self.refresh()

    def refresh(self) -> None:
        p = self.project
        self.items = p.review_items(self.threshold.value(), self.min_frames.value(), self.include_merged.isChecked())
        self.items_changed.emit(self.items)
        self._fill()

    def _mark_edited(self) -> None:
        self._fill()

    def _fill(self) -> None:
        p = self.project
        keep = self.list.currentRow()
        self.list.clear()
        edited = 0
        for it in self.items:
            seg = p.manual[it["start"] : it["end"] + 1]
            is_edited = bool((seg != "").any())
            edited += is_edited
            dur = (it["end"] - it["start"] + 1) / (p.fps or 30)
            text = f"{'[edited] ' if is_edited else ''}{it['kind']}: frames {it['start']} to {it['end']} ({dur:.1f} s)"
            li = QListWidgetItem(text)
            if is_edited:
                li.setForeground(QColor("#8a8f98"))
            self.list.addItem(li)
        if 0 <= keep < self.list.count():
            self.list.setCurrentRow(keep)
        no_conf = len(p.conf) and np.all(np.isnan(p.conf))
        note = " This run has no per-frame confidence (heuristic labels), so only merged-blob stretches appear." if no_conf else ""
        self.status.setText(f"{len(self.items)} items, {edited} already edited.{note}")

    def _jump(self, row: int) -> None:
        if 0 <= row < len(self.items):
            self.list.setCurrentRow(row)
            self.jump_requested.emit(self.items[row]["start"], self.items[row]["end"])

    def step(self, direction: int, from_frame: int | None = None) -> None:
        if not self.items:
            return
        if from_frame is None:
            row = self.list.currentRow()
            row = 0 if row < 0 and direction > 0 else (len(self.items) - 1 if row < 0 else row + direction)
        else:
            if direction > 0:
                row = next((i for i, it in enumerate(self.items) if it["start"] > from_frame), 0)
            else:
                row = next((i for i in range(len(self.items) - 1, -1, -1) if self.items[i]["end"] < from_frame), len(self.items) - 1)
        self._jump(row % len(self.items))


class ZonesPanel(QWidget):
    calibrate_requested = pyqtSignal()
    draw_zone_requested = pyqtSignal()
    clear_calibration_requested = pyqtSignal()

    def __init__(self, project: Project):
        super().__init__()
        self.project = project
        self.scale_label = QLabel("")
        self.scale_label.setWordWrap(True)
        cal = QPushButton("Calibrate arena...")
        cal.clicked.connect(self.calibrate_requested)
        clr = QPushButton("Clear")
        clr.clicked.connect(self.clear_calibration_requested)
        draw = QPushButton("Draw zone...")
        draw.clicked.connect(self.draw_zone_requested)
        rem = QPushButton("Delete zone")
        rem.clicked.connect(self._remove)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Zone", "Animal", "Time (s)", "% of tracked"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        row1, row2 = QHBoxLayout(), QHBoxLayout()
        row1.addWidget(cal)
        row1.addWidget(clr)
        row2.addWidget(draw)
        row2.addWidget(rem)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(self.scale_label)
        lay.addLayout(row1)
        lay.addLayout(row2)
        lay.addWidget(self.table)
        for sig in (project.zones_changed, project.calibration_changed, project.data_changed):
            sig.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        p = self.project
        if p.px_per_cm:
            self.scale_label.setText(f"Scale: {p.px_per_cm:.2f} px per cm. Distances and speeds are in cm.")
        else:
            self.scale_label.setText("No arena calibration. Distances and speeds are in pixels. Calibrate by clicking the four arena corners and entering the real width.")
        z = p.zone_table()
        self.table.setRowCount(len(z))
        for i, r in enumerate(z.itertuples()):
            self.table.setItem(i, 0, _item(r.zone))
            self.table.setItem(i, 1, _item(r.animal))
            self.table.setItem(i, 2, _item(f"{r.seconds:.1f}", True))
            self.table.setItem(i, 3, _item(f"{r.percent:.1f}", True))

    def _remove(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            self.project.remove_zone(self.table.item(row, 0).text())


class AnnotatePanel(QWidget):
    behavior_chosen = pyqtSignal(str)
    paint_toggled = pyqtSignal(bool)

    def __init__(self, project: Project):
        super().__init__()
        self.project = project
        self.active = "sniffing"
        self.buttons: dict[str, QPushButton] = {}
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(6, 6, 6, 6)
        self.lay.addWidget(QLabel("Active behavior (hotkey in brackets)"))
        self.beh_box = QVBoxLayout()
        self.lay.addLayout(self.beh_box)
        self.paint = QPushButton("Paint tool (P): drag on the timeline to add a bout")
        self.paint.setCheckable(True)
        self.paint.toggled.connect(self.paint_toggled)
        self.lay.addWidget(self.paint)
        self.start_btn = QPushButton("Start from model predictions")
        self.clear_btn = QPushButton("Discard all manual edits")
        self.undo_btn = QPushButton("Undo")
        self.redo_btn = QPushButton("Redo")
        self.lay.addWidget(self.start_btn)
        row = QHBoxLayout()
        row.addWidget(self.undo_btn)
        row.addWidget(self.redo_btn)
        self.lay.addLayout(row)
        self.lay.addWidget(self.clear_btn)
        self.bout_label = QLabel("No bout selected. Click a bout in the Final row, or double-click one.")
        self.bout_label.setWordWrap(True)
        self.bout_label.setObjectName("muted")
        self.lay.addWidget(self.bout_label)
        self.rec_label = QLabel("")
        self.rec_label.setWordWrap(True)
        self.lay.addWidget(self.rec_label)
        self.lay.addStretch(1)
        project.data_changed.connect(self._rebuild)
        project.labels_changed.connect(self._sync_buttons)
        self._rebuild()

    def _rebuild(self) -> None:
        while self.beh_box.count():
            w = self.beh_box.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.buttons = {}
        keys = shortcuts.behavior_key_map(self.project.behaviors())
        for b in self.project.behaviors():
            key = keys.get(b, "")
            btn = QPushButton(f"  {b}  [{key}]" if key else f"  {b}")
            btn.setIcon(_icon(b))
            btn.setCheckable(True)
            btn.clicked.connect(lambda _c, name=b: self.set_active(name))
            self.beh_box.addWidget(btn)
            self.buttons[b] = btn
        self.set_active(self.active if self.active in self.buttons else "sniffing", emit=False)
        self._sync_buttons()

    def set_active(self, name: str, emit: bool = True) -> None:
        self.active = name
        for b, btn in self.buttons.items():
            btn.setChecked(b == name)
        if emit:
            self.behavior_chosen.emit(name)

    def _sync_buttons(self) -> None:
        self.undo_btn.setEnabled(self.project.can_undo())
        self.redo_btn.setEnabled(self.project.can_redo())
        self.clear_btn.setEnabled(self.project.n_manual_frames() > 0)

    def show_bout(self, bout: dict | None) -> None:
        if not bout:
            self.bout_label.setText("No bout selected. Click a bout in the Final row, or double-click one.")
            return
        fps = self.project.fps or 30
        dur = (bout["end"] - bout["start"] + 1) / fps
        edited = bool((self.project.manual[bout["start"] : bout["end"] + 1] != "").any())
        self.bout_label.setText(
            f"Selected: {bout['behavior']}, frames {bout['start']} to {bout['end']} ({dur:.2f} s)" + (", edited" if edited else "")
        )

    def show_recording(self, text: str) -> None:
        self.rec_label.setText(text)


class _ImageDialog(QDialog):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setObjectName("muted")
        save = QPushButton("Save as PNG...")
        save.clicked.connect(self._save)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(save)
        row.addWidget(close)
        lay = QVBoxLayout(self)
        lay.addWidget(self.image_label, 1)
        lay.addWidget(self.note)
        lay.addLayout(row)
        self._qimage: QImage | None = None

    def set_bgr(self, bgr: np.ndarray, max_w: int = 760) -> None:
        h, w = bgr.shape[:2]
        scale = min(1.0, max_w / w)
        shown = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else bgr
        rgb = cv2.cvtColor(np.ascontiguousarray(shown), cv2.COLOR_BGR2RGB)
        self._qimage = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format.Format_RGB888).copy()
        self._full = bgr
        self.image_label.setPixmap(QPixmap.fromImage(self._qimage))

    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save image", "", "PNG (*.png)")
        if path and getattr(self, "_full", None) is not None:
            cv2.imwrite(path, self._full)


class HeatmapDialog(_ImageDialog):
    def __init__(self, project: Project, background: np.ndarray | None, parent=None):
        super().__init__("Occupancy heatmap", parent)
        heat = analysis.occupancy_heatmap(project.positions, (project.height, project.width), bins=(72, 96), sigma=2.0)
        color = cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        color = cv2.resize(color, (project.width, project.height), interpolation=cv2.INTER_CUBIC)
        base = background if background is not None else np.zeros((project.height, project.width, 3), np.uint8)
        alpha = cv2.resize(np.clip(heat * 1.4, 0, 0.8), (project.width, project.height), interpolation=cv2.INTER_CUBIC)[..., None]
        out = (base * (1 - alpha) + color * alpha).astype(np.uint8)
        for name, poly in project.zones.items():
            cv2.polylines(out, [np.asarray(poly, np.int32).reshape(-1, 1, 2)], True, (180, 105, 255), 2, cv2.LINE_AA)
        self.set_bgr(out)
        self.note.setText("Brighter means more time spent there (both animals combined). Zones are outlined in pink.")


class TransitionView(QWidget):
    def __init__(self, behaviors: list[str], counts: np.ndarray, probs: np.ndarray):
        super().__init__()
        self.behaviors, self.counts, self.probs = behaviors, counts, probs
        self.setMinimumSize(420, 380)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        n = len(self.behaviors)
        if n == 0:
            return
        cx, cy = self.width() / 2, self.height() / 2
        radius = min(self.width(), self.height()) / 2 - 70
        pts = [QPointF(cx + radius * math.cos(2 * math.pi * i / n - math.pi / 2), cy + radius * math.sin(2 * math.pi * i / n - math.pi / 2)) for i in range(n)]
        node_r = 28
        pal = self.palette()
        for i in range(n):
            for j in range(n):
                if self.counts[i, j] <= 0:
                    continue
                prob = self.probs[i, j]
                col = QColor(*behavior_rgb(self.behaviors[i]))
                col.setAlpha(190)
                p.setPen(QPen(col, 1.0 + 6 * prob))
                if i == j:
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    out = QPointF(pts[i].x() - (cx - pts[i].x()) * 0.0, pts[i].y())
                    d = QPointF(pts[i].x() - cx, pts[i].y() - cy)
                    norm = math.hypot(d.x(), d.y()) or 1
                    c = QPointF(pts[i].x() + d.x() / norm * (node_r + 8), pts[i].y() + d.y() / norm * (node_r + 8))
                    p.drawEllipse(c, 14, 14)
                    p.setPen(pal.text().color())
                    p.drawText(c + QPointF(-14, -18), f"{prob:.0%} ({int(self.counts[i, j])})")
                    del out
                    continue
                a, b = pts[i], pts[j]
                vx, vy = b.x() - a.x(), b.y() - a.y()
                length = math.hypot(vx, vy) or 1
                ux, uy = vx / length, vy / length
                off = QPointF(-uy * 7, ux * 7)
                start = QPointF(a.x() + ux * node_r + off.x(), a.y() + uy * node_r + off.y())
                end = QPointF(b.x() - ux * (node_r + 6) + off.x(), b.y() - uy * (node_r + 6) + off.y())
                p.drawLine(start, end)
                head = QPointF(end.x() - ux * 9, end.y() - uy * 9)
                p.setBrush(col)
                p.drawPolygon([end, QPointF(head.x() - uy * 5, head.y() + ux * 5), QPointF(head.x() + uy * 5, head.y() - ux * 5)])
                mid = QPointF((start.x() + end.x()) / 2, (start.y() + end.y()) / 2)
                p.setPen(pal.text().color())
                p.drawText(mid + QPointF(4 * (1 if off.x() >= 0 else -4), 4), f"{prob:.0%} ({int(self.counts[i, j])})")
        for i, b in enumerate(self.behaviors):
            p.setPen(QPen(QColor("#ffffff"), 1.5))
            p.setBrush(QColor(*behavior_rgb(b)))
            p.drawEllipse(pts[i], node_r, node_r)
            p.setPen(QColor("#ffffff"))
            f = QFont(self.font())
            f.setBold(True)
            p.setFont(f)
            p.drawText(QRectF(pts[i].x() - node_r, pts[i].y() - node_r, 2 * node_r, 2 * node_r), Qt.AlignmentFlag.AlignCenter, b)


class TransitionDialog(QDialog):
    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Behavior transitions")
        behaviors = [b for b in project.behaviors() if b != BACKGROUND]
        counts, probs = analysis.transition_matrix(project.bouts(), behaviors)
        lay = QVBoxLayout(self)
        lay.addWidget(TransitionView(behaviors, counts, probs), 1)
        table = QTableWidget(len(behaviors), len(behaviors))
        table.setHorizontalHeaderLabels([f"then {b}" for b in behaviors])
        table.setVerticalHeaderLabels([f"after {b}" for b in behaviors])
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        for i in range(len(behaviors)):
            for j in range(len(behaviors)):
                it = _item(f"{probs[i, j]:.0%}  (n={int(counts[i, j])})" if counts[i, j] else "-", True)
                if counts[i, j]:
                    it.setBackground(QColor(77, 141, 247, int(40 + 180 * probs[i, j])))
                table.setItem(i, j, it)
        table.setMaximumHeight(34 * (len(behaviors) + 2))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        lay.addWidget(table)
        note = QLabel(
            f"Counts consecutive bouts in the final labels ({len(project.bouts())} bouts). The percent is how often the row's "
            "behavior was followed by the column's, so each row sums to 100% (rows with no following bout are blank). "
            "With few bouts these percentages are noisy."
        )
        note.setWordWrap(True)
        note.setObjectName("muted")
        lay.addWidget(note)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        lay.addWidget(close)
        self.resize(560, 640)


class ShortcutsDialog(QDialog):
    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard shortcuts")
        lay = QVBoxLayout(self)
        view = QTableWidget(0, 3)
        view.setHorizontalHeaderLabels(["Group", "Keys", "Action"])
        view.verticalHeader().setVisible(False)
        view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        rows = [(cat, key, desc) for _id, key, cat, desc in shortcuts.SHORTCUTS]
        for beh, key in shortcuts.behavior_key_map(project.behaviors()).items():
            rows.append(("Behavior keys", key, f"Label / record {beh}"))
        rows += [(g, k, d) for g, k, d in shortcuts.MOUSE_HELP]
        view.setRowCount(len(rows))
        for i, (g, k, d) in enumerate(rows):
            view.setItem(i, 0, _item(g))
            view.setItem(i, 1, _item(k))
            view.setItem(i, 2, _item(d))
        view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        view.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(view)
        note = QLabel(shortcuts.RECORDING_HELP)
        note.setWordWrap(True)
        note.setObjectName("muted")
        lay.addWidget(note)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        lay.addWidget(close)
        self.resize(640, 640)
