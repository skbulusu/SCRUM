from __future__ import annotations

import time
import traceback
from pathlib import Path

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from scrum.app import export
from scrum.app.inference import run_scrum_detection_full
from scrum.app.project import DEFAULT_BEHAVIORS, Project
from scrum.app.results import DetectionResult, find_cache, results_dir_for

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
REPORT_BEHAVIORS = [b for b in DEFAULT_BEHAVIORS if b != "none"]


def find_videos(folder: str | Path) -> list[str]:
    return sorted(str(p) for p in Path(folder).rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS and "scrum_results" not in p.parts)


def process_video(video_path: str, use_cache: bool = True, progress=None, out_dir: str | Path | None = None) -> dict:
    t0 = time.time()
    out = Path(out_dir) if out_dir else results_dir_for(video_path)
    cached = find_cache(video_path) if use_cache else None
    result = None
    from_cache = False
    if cached is not None:
        try:
            r = DetectionResult.load(cached)
            if r.matches_video(video_path) and r.mode == "scrum":
                result, from_cache = r, True
        except Exception:
            result = None
    cancelled = False
    if result is None:
        def cb(done, total):
            nonlocal cancelled
            if progress is not None and progress(done, total) is False:
                cancelled = True
                return False
            return None

        result = run_scrum_detection_full(video_path, progress=cb)
    if cancelled:
        return {"video": video_path, "status": "cancelled"}

    project = Project()
    project.set_video(video_path, result.fps, result.n_frames, result.width, result.height)
    project.set_result(result)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem
    if not from_cache:
        result.save(out / f"{stem}.scrum_cache.npz")
    export.export_labels_csv(project, out / f"{stem}_labels.csv")
    export.export_bouts_csv(project, out / f"{stem}_bouts.csv")
    export.export_summary_csv(project, out / f"{stem}_summary.csv")

    summary = project.summary().set_index("behavior")
    row = {
        "video": video_path,
        "status": "done (cached)" if from_cache else "done",
        "frames": project.n_frames,
        "duration_s": round(project.n_frames / project.fps, 2),
        "bouts": len(project.bouts()),
        "merged_pct": round(100.0 * float(project.merged.mean()) if len(project.merged) else 0.0, 1),
        "source": project.source,
        "seconds": round(time.time() - t0, 1),
    }
    for b in REPORT_BEHAVIORS:
        row[f"{b}_s"] = round(float(summary.loc[b, "seconds"]), 2) if b in summary.index else 0.0
    return row


class BatchWorker(QObject):
    progress = pyqtSignal(int, int, int)
    row_done = pyqtSignal(int, object)
    finished = pyqtSignal()

    def __init__(self, videos: list[str], use_cache: bool):
        super().__init__()
        self.videos, self.use_cache = videos, use_cache
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        for i, v in enumerate(self.videos):
            if self._cancel:
                self.row_done.emit(i, {"video": v, "status": "cancelled"})
                continue
            try:
                row = process_video(v, self.use_cache, progress=lambda d, t, i=i: (self.progress.emit(i, d, t), not self._cancel)[1])
            except Exception as e:
                row = {"video": v, "status": f"failed: {e}", "trace": traceback.format_exc()}
            self.row_done.emit(i, row)
        self.finished.emit()


class BatchWindow(QWidget):
    open_requested = pyqtSignal(str)

    COLS = ["Video", "Status", "Progress", "Duration (s)"] + [f"{b} (s)" for b in REPORT_BEHAVIORS] + ["Bouts", "Merged %"]

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("SCRUM batch queue")
        self.resize(980, 520)
        self.setAcceptDrops(True)
        self.videos: list[str] = []
        self.rows: dict[int, dict] = {}
        self._thread: QThread | None = None
        self._worker: BatchWorker | None = None

        add_folder = QPushButton("Add folder...")
        add_folder.clicked.connect(self._add_folder)
        add_files = QPushButton("Add videos...")
        add_files.clicked.connect(self._add_files)
        clear = QPushButton("Clear list")
        clear.clicked.connect(self._clear)
        self.cache_box = QCheckBox("Reuse earlier results when available")
        self.cache_box.setChecked(True)
        self.run_btn = QPushButton("Run queue")
        self.run_btn.clicked.connect(self.run)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        self.export_btn = QPushButton("Save results table...")
        self.export_btn.clicked.connect(self._save_table)
        top = QHBoxLayout()
        for w in (add_folder, add_files, clear):
            top.addWidget(w)
        top.addStretch(1)
        top.addWidget(self.cache_box)
        top.addWidget(self.run_btn)
        top.addWidget(self.cancel_btn)

        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(2, 150)
        self.table.cellDoubleClicked.connect(self._open_row)

        self.status = QLabel("Drop a folder or videos here, or use the buttons. Results are written to a scrum_results folder beside each video.")
        self.status.setObjectName("muted")
        self.status.setWordWrap(True)
        open_btn = QPushButton("Open selected in viewer")
        open_btn.clicked.connect(lambda: self._open_row(self.table.currentRow(), 0))
        bottom = QHBoxLayout()
        bottom.addWidget(self.status, 1)
        bottom.addWidget(open_btn)
        bottom.addWidget(self.export_btn)
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.table, 1)
        lay.addLayout(bottom)


    def add_paths(self, paths: list[str]) -> int:
        added = 0
        for p in paths:
            found = find_videos(p) if Path(p).is_dir() else ([p] if Path(p).suffix.lower() in VIDEO_EXTENSIONS else [])
            for v in found:
                if v not in self.videos:
                    self.videos.append(v)
                    added += 1
        self._refill()
        return added

    def _add_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Folder of videos")
        if d:
            n = self.add_paths([d])
            if n == 0:
                QMessageBox.information(self, "No videos found", "No .mp4, .avi, .mov, .mkv or .m4v files were found in that folder.")

    def _add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Videos", "", "Video files (*.mp4 *.avi *.mov *.mkv *.m4v)")
        self.add_paths(files)

    def _clear(self) -> None:
        if self._thread is None:
            self.videos, self.rows = [], {}
            self._refill()

    def dragEnterEvent(self, ev) -> None:
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev) -> None:
        self.add_paths([u.toLocalFile() for u in ev.mimeData().urls() if u.isLocalFile()])


    def _refill(self) -> None:
        self.table.setRowCount(len(self.videos))
        for i, v in enumerate(self.videos):
            row = self.rows.get(i, {})
            self._set_text(i, 0, Path(v).name)
            self.table.item(i, 0).setToolTip(v)
            self._set_text(i, 1, row.get("status", "queued"))
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(100 if str(row.get("status", "")).startswith("done") else 0)
            self.table.setCellWidget(i, 2, bar)
            self._fill_results(i, row)
        self.status.setText(f"{len(self.videos)} videos queued." if self.videos else self.status.text())

    def _set_text(self, r: int, c: int, text: str) -> None:
        it = QTableWidgetItem(text)
        if c >= 3:
            it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.table.setItem(r, c, it)

    def _fill_results(self, r: int, row: dict) -> None:
        if not row or not str(row.get("status", "")).startswith("done"):
            return
        self._set_text(r, 3, f"{row['duration_s']:.1f}")
        for k, b in enumerate(REPORT_BEHAVIORS):
            self._set_text(r, 4 + k, f"{row.get(f'{b}_s', 0):.1f}")
        self._set_text(r, 4 + len(REPORT_BEHAVIORS), str(row["bouts"]))
        self._set_text(r, 5 + len(REPORT_BEHAVIORS), f"{row['merged_pct']:.1f}")


    def run(self) -> None:
        if self._thread is not None or not self.videos:
            return
        self.rows = {}
        self._refill()
        self._worker = BatchWorker(list(self.videos), self.cache_box.isChecked())
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.row_done.connect(self._on_row_done)
        self._worker.finished.connect(self._on_finished)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.status.setText("Running...")
        self._thread.start()

    def _cancel(self) -> None:
        if self._worker:
            self._worker.cancel()
            self.status.setText("Cancelling after the current video reports progress...")

    def _on_progress(self, row: int, done: int, total: int) -> None:
        bar = self.table.cellWidget(row, 2)
        if bar is not None and total > 0:
            bar.setValue(int(100 * done / total))
        self._set_text(row, 1, "running")

    def _on_row_done(self, row: int, data: dict) -> None:
        self.rows[row] = data
        self._set_text(row, 1, data.get("status", ""))
        bar = self.table.cellWidget(row, 2)
        if bar is not None and str(data.get("status", "")).startswith("done"):
            bar.setValue(100)
        self._fill_results(row, data)
        if str(data.get("status", "")).startswith("failed"):
            self.table.item(row, 1).setToolTip(data.get("trace", ""))

    def _on_finished(self) -> None:
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        self._thread, self._worker = None, None
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        ok = sum(1 for r in self.rows.values() if str(r.get("status", "")).startswith("done"))
        self.status.setText(f"Finished: {ok} of {len(self.videos)} videos processed. Files are in each video's scrum_results folder.")

    def _open_row(self, row: int, _col: int) -> None:
        if 0 <= row < len(self.videos):
            self.open_requested.emit(self.videos[row])

    def _save_table(self) -> None:
        import pandas as pd

        rows = [r for r in self.rows.values() if "frames" in r]
        if not rows:
            QMessageBox.information(self, "Nothing to save", "Run the queue first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save results table", "scrum_batch_results.csv", "CSV (*.csv)")
        if path:
            pd.DataFrame(rows).drop(columns=["trace"], errors="ignore").to_csv(path, index=False)

    def closeEvent(self, ev) -> None:
        if self._thread is not None:
            self._cancel()
        ev.accept()
