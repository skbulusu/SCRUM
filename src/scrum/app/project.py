from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6.QtCore import QObject, pyqtSignal

from scrum.app import analysis
from scrum.app.analysis import BACKGROUND, Calibration
from scrum.app.results import DetectionResult

DEFAULT_BEHAVIORS = ["none", "sniffing", "biting", "mounting"]
SESSION_VERSION = 1
UNDO_LIMIT = 500


class _Edit:
    __slots__ = ("start", "old", "new", "text")

    def __init__(self, start: int, old: np.ndarray, new: np.ndarray, text: str):
        self.start, self.old, self.new, self.text = start, old, new, text


class Project(QObject):
    data_changed = pyqtSignal()
    labels_changed = pyqtSignal()
    bookmarks_changed = pyqtSignal()
    calibration_changed = pyqtSignal()
    zones_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.video_path: str | None = None
        self.fps = 30.0
        self.n_frames = 0
        self.width = 0
        self.height = 0
        self.mode = "scrum"
        self.source = ""

        self.tracks = pd.DataFrame(columns=["frame", "track_id", "keypoint", "x", "y"])
        self.tracks_lookup: dict[int, list[tuple]] = {}
        self.outlines: dict[int, list[dict]] = {}
        self.merged = np.zeros(0, dtype=bool)

        self.pred = np.zeros(0, dtype=object)
        self.conf = np.zeros(0, dtype=float)
        self.manual = np.zeros(0, dtype=object)
        self._final: np.ndarray | None = None
        self._undo: list[_Edit] = []
        self._redo: list[_Edit] = []

        self.bookmarks: list[dict] = []
        self.calibration: Calibration | None = None
        self.zones: dict[str, list[tuple[float, float]]] = {}

        self.positions: dict[str, np.ndarray] = {}
        self.speed: dict[str, np.ndarray] = {}
        self.distance = np.zeros(0)


    @property
    def px_per_cm(self) -> float | None:
        return self.calibration.px_per_cm if self.calibration else None

    @property
    def units(self) -> str:
        return "cm" if self.px_per_cm else "px"

    @property
    def has_video(self) -> bool:
        return self.video_path is not None and self.n_frames > 0

    @property
    def has_predictions(self) -> bool:
        return bool(len(self.pred)) and any(p != "" for p in self.pred)

    def behaviors(self) -> list[str]:
        found = [b for b in dict.fromkeys(list(self.pred) + list(self.manual)) if b not in ("", None)]
        return DEFAULT_BEHAVIORS + [b for b in found if b not in DEFAULT_BEHAVIORS]

    def animals(self) -> list[str]:
        return list(self.positions)


    def set_video(self, path: str, fps: float, n_frames: int, width: int, height: int) -> None:
        self.video_path, self.fps, self.n_frames, self.width, self.height = path, fps or 30.0, n_frames, width, height
        self.source = ""
        self.tracks = pd.DataFrame(columns=["frame", "track_id", "keypoint", "x", "y"])
        self.tracks_lookup = {}
        self.outlines = {}
        self.merged = np.zeros(n_frames, dtype=bool)
        self.pred = np.full(n_frames, "", dtype=object)
        self.conf = np.full(n_frames, np.nan)
        self.manual = np.full(n_frames, "", dtype=object)
        self._final = None
        self._undo.clear()
        self._redo.clear()
        self.bookmarks = []
        self.calibration = None
        self.zones = {}
        self._recompute_motion()
        self.data_changed.emit()
        self.labels_changed.emit()
        self.bookmarks_changed.emit()
        self.calibration_changed.emit()
        self.zones_changed.emit()

    def clear_detection(self) -> None:
        n = self.n_frames
        self.tracks = pd.DataFrame(columns=["frame", "track_id", "keypoint", "x", "y"])
        self.tracks_lookup = {}
        self.outlines = {}
        self.merged = np.zeros(n, dtype=bool)
        self.pred = np.full(n, "", dtype=object)
        self.conf = np.full(n, np.nan)
        self.source = ""
        self._final = None
        self._recompute_motion()
        self.data_changed.emit()
        self.labels_changed.emit()

    def set_result(self, result: DetectionResult) -> None:
        self.mode = result.mode
        self.source = result.source
        self._set_tracks(result.tracks)
        self.outlines = result.outlines or {}
        merged = np.zeros(self.n_frames, dtype=bool)
        if result.merged is not None:
            m = np.asarray(result.merged, dtype=bool)[: self.n_frames]
            merged[: len(m)] = m
        self.merged = merged
        self._set_predictions(result.predictions)
        self.data_changed.emit()
        self.labels_changed.emit()

    def set_tracks(self, tracks: pd.DataFrame) -> None:
        self._set_tracks(tracks)
        self.outlines = {}
        self.data_changed.emit()

    def set_predictions(self, predictions: pd.DataFrame, source: str | None = None) -> None:
        if source is not None:
            self.source = source
        self._set_predictions(predictions)
        self.labels_changed.emit()

    def _set_tracks(self, tracks: pd.DataFrame) -> None:
        self.tracks = tracks[["frame", "track_id", "keypoint", "x", "y"]].copy() if len(tracks) else tracks
        lookup: dict[int, list[tuple]] = defaultdict(list)
        for f, tid, kp, x, y in zip(
            self.tracks["frame"].to_numpy(), self.tracks["track_id"].to_numpy(), self.tracks["keypoint"].to_numpy(),
            self.tracks["x"].to_numpy(), self.tracks["y"].to_numpy(),
        ):
            lookup[int(f)].append((str(tid), str(kp), float(x), float(y)))
        self.tracks_lookup = dict(lookup)
        self._recompute_motion()

    def _set_predictions(self, predictions: pd.DataFrame) -> None:
        n = self.n_frames or (int(predictions["frame"].max()) + 1 if len(predictions) else 0)
        if self.n_frames == 0:
            self.n_frames = n
            self.manual = np.full(n, "", dtype=object)
            self.merged = np.zeros(n, dtype=bool)
        pred = np.full(n, "", dtype=object)
        conf = np.full(n, np.nan)
        if len(predictions):
            f = predictions["frame"].to_numpy().astype(int)
            ok = (f >= 0) & (f < n)
            pred[f[ok]] = predictions["behavior"].astype(str).to_numpy()[ok]
            if "confidence" in predictions.columns:
                conf[f[ok]] = predictions["confidence"].to_numpy(dtype=float)[ok]
        self.pred, self.conf = pred, conf
        self._final = None

    def _recompute_motion(self) -> None:
        n = self.n_frames
        animals = analysis.pick_animals(self.tracks)
        if not animals or n == 0:
            self.positions, self.speed, self.distance = {}, {}, np.full(n, np.nan)
            return
        self.positions = analysis.positions_from_tracks(self.tracks, n, animals)
        self._recompute_scale_dependent()

    def _recompute_scale_dependent(self) -> None:
        if not self.positions:
            return
        self.speed = analysis.speeds(self.positions, self.fps, self.px_per_cm)
        filled = {a: analysis.fill_small_gaps(p) for a, p in self.positions.items()}
        self.distance = analysis.pair_distance(filled, self.px_per_cm)


    def final_labels(self) -> np.ndarray:
        if self._final is None:
            final = np.where(self.manual != "", self.manual, self.pred)
            self._final = np.where(final == "", BACKGROUND, final).astype(object) if len(final) else final
        return self._final

    def label_at(self, frame: int) -> str | None:
        if 0 <= frame < len(self.pred):
            return self.final_labels()[frame]
        return None

    def bouts(self) -> list[dict]:
        return analysis.bouts_from_labels(self.final_labels())

    def pred_bouts(self) -> list[dict]:
        return analysis.bouts_from_labels(np.where(self.pred == "", BACKGROUND, self.pred))

    def bout_at(self, frame: int) -> dict | None:
        for b in self.bouts():
            if b["start"] <= frame <= b["end"]:
                return b
        return None

    def n_manual_frames(self) -> int:
        return int((self.manual != "").sum())

    def disagreement_mask(self) -> np.ndarray:
        pred = np.where(self.pred == "", BACKGROUND, self.pred)
        return self.final_labels() != pred


    def _apply(self, start: int, end: int, values, text: str, record: bool = True) -> bool:
        start, end = max(0, int(start)), min(self.n_frames - 1, int(end))
        if end < start:
            return False
        old = self.manual[start : end + 1].copy()
        new = np.empty(end - start + 1, dtype=object)
        new[:] = values
        if np.array_equal(old, new):
            return False
        self.manual[start : end + 1] = new
        self._final = None
        if record:
            self._undo.append(_Edit(start, old, new.copy(), text))
            if len(self._undo) > UNDO_LIMIT:
                self._undo.pop(0)
            self._redo.clear()
        self.labels_changed.emit()
        return True

    def set_range(self, start: int, end: int, behavior: str) -> bool:
        if start > end:
            start, end = end, start
        return self._apply(start, end, behavior, f"label {behavior}")

    def add_bout(self, start: int, end: int, behavior: str) -> bool:
        return self.set_range(start, end, behavior)

    def delete_bout(self, bout: dict) -> bool:
        return self._apply(bout["start"], bout["end"], BACKGROUND, f"delete {bout['behavior']} bout")

    def relabel_bout(self, bout: dict, behavior: str) -> bool:
        return self._apply(bout["start"], bout["end"], behavior, f"relabel as {behavior}")

    def resize_bout(self, bout: dict, new_start: int, new_end: int) -> bool:
        new_start, new_end = max(0, min(new_start, new_end)), min(self.n_frames - 1, max(new_start, new_end))
        lo, hi = min(bout["start"], new_start), max(bout["end"], new_end)
        values = self.final_labels()[lo : hi + 1].copy()
        values[:] = np.where(
            (np.arange(lo, hi + 1) >= bout["start"]) & (np.arange(lo, hi + 1) <= bout["end"]), BACKGROUND, values
        )
        values[new_start - lo : new_end - lo + 1] = bout["behavior"]
        return self._apply(lo, hi, values, f"resize {bout['behavior']} bout")

    def split_bout(self, frame: int, gap: int = 1) -> bool:
        b = self.bout_at(frame)
        if b is None or b["end"] - b["start"] < 2:
            return False
        s = max(b["start"] + 1, min(frame, b["end"] - gap))
        return self._apply(s, s + gap - 1, BACKGROUND, f"split {b['behavior']} bout")

    def merge_bouts(self, first: dict, second: dict) -> bool:
        a, b = sorted([first, second], key=lambda x: x["start"])
        return self._apply(a["start"], b["end"], a["behavior"], f"merge {a['behavior']} bouts")

    def merge_with_next(self, bout: dict) -> bool:
        nxt = next((b for b in self.bouts() if b["start"] > bout["end"]), None)
        return self.merge_bouts(bout, nxt) if nxt else False

    def start_manual_from_predictions(self) -> bool:
        pred = np.where(self.pred == "", BACKGROUND, self.pred)
        return self._apply(0, self.n_frames - 1, pred, "start from predictions")

    def clear_manual(self) -> bool:
        return self._apply(0, self.n_frames - 1, "", "clear manual labels")

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> str | None:
        if not self._undo:
            return None
        e = self._undo.pop()
        self.manual[e.start : e.start + len(e.old)] = e.old
        self._final = None
        self._redo.append(e)
        self.labels_changed.emit()
        return e.text

    def redo(self) -> str | None:
        if not self._redo:
            return None
        e = self._redo.pop()
        self.manual[e.start : e.start + len(e.new)] = e.new
        self._final = None
        self._undo.append(e)
        self.labels_changed.emit()
        return e.text


    def add_bookmark(self, frame: int, note: str = "") -> None:
        self.bookmarks = [b for b in self.bookmarks if b["frame"] != frame]
        self.bookmarks.append({"frame": int(frame), "note": note})
        self.bookmarks.sort(key=lambda b: b["frame"])
        self.bookmarks_changed.emit()

    def remove_bookmark(self, frame: int) -> None:
        self.bookmarks = [b for b in self.bookmarks if b["frame"] != frame]
        self.bookmarks_changed.emit()

    def bookmark_near(self, frame: int, tolerance: int = 0) -> dict | None:
        return next((b for b in self.bookmarks if abs(b["frame"] - frame) <= tolerance), None)


    def set_calibration(self, calibration: Calibration | None) -> None:
        self.calibration = calibration
        self._recompute_scale_dependent()
        self.calibration_changed.emit()
        self.data_changed.emit()

    def add_zone(self, name: str, polygon: list[tuple[float, float]]) -> None:
        self.zones[name] = [(float(x), float(y)) for x, y in polygon]
        self.zones_changed.emit()

    def remove_zone(self, name: str) -> None:
        self.zones.pop(name, None)
        self.zones_changed.emit()


    def review_items(self, threshold: float = 0.6, min_frames: int = 3, include_merged: bool = True) -> list[dict]:
        items = [
            {"start": s, "end": e, "kind": "low confidence"}
            for s, e in analysis.low_confidence_stretches(self.conf, threshold, min_frames)
        ]
        if include_merged and self.merged.any():
            items += [{"start": s, "end": e, "kind": "merged blob"} for s, e in analysis.runs_of_mask(self.merged) if e - s + 1 >= min_frames]
        items.sort(key=lambda i: i["start"])
        return items


    def summary(self) -> pd.DataFrame:
        return analysis.behavior_summary(self.final_labels(), self.fps, self.behaviors())

    def zone_table(self) -> pd.DataFrame:
        return analysis.zone_occupancy(self.positions, self.zones, self.fps) if self.zones else pd.DataFrame(
            columns=["zone", "animal", "frames", "seconds", "percent"]
        )

    def to_result(self) -> DetectionResult:
        n = self.n_frames
        preds = pd.DataFrame(
            {"frame": np.arange(n), "behavior": np.where(self.pred == "", BACKGROUND, self.pred), "confidence": self.conf}
        )
        return DetectionResult(
            video_path=self.video_path or "", fps=self.fps, n_frames=n, width=self.width, height=self.height,
            source=self.source, mode=self.mode, tracks=self.tracks, predictions=preds, outlines=self.outlines, merged=self.merged,
        )


    def manual_runs(self) -> list[dict]:
        runs, n, i = [], len(self.manual), 0
        while i < n:
            v = self.manual[i]
            j = i
            while j + 1 < n and self.manual[j + 1] == v:
                j += 1
            if v != "":
                runs.append({"start": i, "end": j, "behavior": str(v)})
            i = j + 1
        return runs

    def to_session_dict(self, cache_name: str | None) -> dict:
        return {
            "version": SESSION_VERSION,
            "video_path": self.video_path,
            "mode": self.mode,
            "cache": cache_name,
            "manual": self.manual_runs(),
            "bookmarks": self.bookmarks,
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "zones": {k: [list(p) for p in v] for k, v in self.zones.items()},
        }

    def apply_session_dict(self, d: dict) -> None:
        self.manual = np.full(self.n_frames, "", dtype=object)
        for r in d.get("manual", []):
            self.manual[max(0, r["start"]) : min(self.n_frames, r["end"] + 1)] = r["behavior"]
        self._final = None
        self._undo.clear()
        self._redo.clear()
        self.bookmarks = sorted(d.get("bookmarks", []), key=lambda b: b["frame"])
        self.calibration = Calibration.from_dict(d["calibration"]) if d.get("calibration") else None
        self.zones = {k: [tuple(p) for p in v] for k, v in (d.get("zones") or {}).items()}
        self._recompute_scale_dependent()
        self.labels_changed.emit()
        self.bookmarks_changed.emit()
        self.calibration_changed.emit()
        self.zones_changed.emit()
        self.data_changed.emit()

    def save_session(self, path: str | Path, extra: dict | None = None) -> None:
        path = Path(path)
        cache = path.with_suffix(".scrum_cache.npz")
        self.to_result().save(cache)
        d = self.to_session_dict(cache.name)
        d.update(extra or {})
        path.write_text(json.dumps(d, indent=1))

    @staticmethod
    def read_session(path: str | Path) -> dict:
        d = json.loads(Path(path).read_text())
        if d.get("version") != SESSION_VERSION:
            raise ValueError("this session file was written by a different version of SCRUM")
        return d
