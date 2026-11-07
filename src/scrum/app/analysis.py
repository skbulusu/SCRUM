from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd

BACKGROUND = "none"
MERGED_TRACK_ID = "rat_0+rat_1"


def bouts_from_labels(labels, background: str = BACKGROUND) -> list[dict]:
    labels = np.asarray(labels, dtype=object)
    bouts = []
    n = len(labels)
    i = 0
    while i < n:
        lab = labels[i]
        j = i
        while j + 1 < n and labels[j + 1] == lab:
            j += 1
        if lab not in (background, "", None):
            bouts.append({"start": int(i), "end": int(j), "behavior": str(lab)})
        i = j + 1
    return bouts


def labels_from_bouts(bouts: list[dict], n_frames: int, background: str = BACKGROUND) -> np.ndarray:
    labels = np.full(n_frames, background, dtype=object)
    for b in bouts:
        labels[max(0, b["start"]) : min(n_frames, b["end"] + 1)] = b["behavior"]
    return labels


def behavior_summary(labels, fps: float, behaviors: list[str] | None = None, background: str = BACKGROUND) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=object)
    n = max(1, len(labels))
    fps = fps or 30.0
    found = [b for b in dict.fromkeys(labels.tolist()) if b not in ("", None)]
    behaviors = list(behaviors) if behaviors is not None else found
    for b in found:
        if b not in behaviors:
            behaviors.append(b)
    bouts = bouts_from_labels(labels, background=background)
    rows = []
    for beh in behaviors:
        if beh == background:
            continue
        mine = [b for b in bouts if b["behavior"] == beh]
        frames = int((labels == beh).sum())
        lengths = [(b["end"] - b["start"] + 1) / fps for b in mine]
        rows.append(
            {
                "behavior": beh,
                "frames": frames,
                "seconds": frames / fps,
                "percent": 100.0 * frames / n,
                "bouts": len(mine),
                "mean_bout_s": float(np.mean(lengths)) if lengths else 0.0,
                "longest_bout_s": float(np.max(lengths)) if lengths else 0.0,
                "latency_s": mine[0]["start"] / fps if mine else float("nan"),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["behavior", "frames", "seconds", "percent", "bouts", "mean_bout_s", "longest_bout_s", "latency_s"],
    )


def transition_matrix(bouts: list[dict], behaviors: list[str]) -> tuple[np.ndarray, np.ndarray]:
    idx = {b: i for i, b in enumerate(behaviors)}
    counts = np.zeros((len(behaviors), len(behaviors)))
    ordered = sorted(bouts, key=lambda b: b["start"])
    for a, b in zip(ordered[:-1], ordered[1:]):
        if a["behavior"] in idx and b["behavior"] in idx:
            counts[idx[a["behavior"]], idx[b["behavior"]]] += 1
    totals = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    return counts, probs


def runs_of_mask(mask) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    d = np.diff(padded.astype(int))
    starts = np.nonzero(d == 1)[0]
    ends = np.nonzero(d == -1)[0] - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def low_confidence_stretches(confidence, threshold: float = 0.6, min_frames: int = 3) -> list[tuple[int, int]]:
    conf = np.asarray(confidence, dtype=float)
    mask = np.where(np.isnan(conf), False, conf < threshold)
    return [(s, e) for s, e in runs_of_mask(mask) if e - s + 1 >= min_frames]


def pick_animals(tracks: pd.DataFrame, max_animals: int = 2) -> list[str]:
    if tracks is None or tracks.empty:
        return []
    counts = tracks[tracks["track_id"] != MERGED_TRACK_ID].groupby("track_id")["frame"].nunique()
    return sorted(counts.nlargest(max_animals).index.tolist())


def positions_from_tracks(tracks: pd.DataFrame, n_frames: int, animals: list[str] | None = None) -> dict[str, np.ndarray]:
    animals = animals or pick_animals(tracks)
    out = {a: np.full((n_frames, 2), np.nan) for a in animals}
    if tracks is None or tracks.empty:
        return out
    grouped = tracks.groupby(["track_id", "frame"])[["x", "y"]].mean()
    for a in animals:
        if a in grouped.index.get_level_values(0):
            sub = grouped.loc[a]
            f = sub.index.to_numpy()
            ok = (f >= 0) & (f < n_frames)
            out[a][f[ok]] = sub.to_numpy()[ok]
    if MERGED_TRACK_ID in set(tracks["track_id"]):
        merged = tracks[tracks["track_id"] == MERGED_TRACK_ID].groupby("frame")[["x", "y"]].mean()
        f = merged.index.to_numpy()
        ok = (f >= 0) & (f < n_frames)
        for a in animals:
            gap = np.isnan(out[a][f[ok], 0])
            idx = f[ok][gap]
            out[a][idx] = merged.to_numpy()[ok][gap]
    return out


def fill_small_gaps(pos: np.ndarray, max_gap: int = 8) -> np.ndarray:
    out = pos.copy()
    for c in range(out.shape[1]):
        col = out[:, c]
        isn = np.isnan(col)
        if not isn.any() or isn.all():
            continue
        idx = np.arange(len(col))
        filled = np.interp(idx, idx[~isn], col[~isn])
        for s, e in runs_of_mask(isn):
            if e - s + 1 <= max_gap and s > 0 and e < len(col) - 1:
                col[s : e + 1] = filled[s : e + 1]
        out[:, c] = col
    return out


def pair_distance(positions: dict[str, np.ndarray], px_per_cm: float | None = None) -> np.ndarray:
    ids = sorted(positions)
    n = len(next(iter(positions.values()))) if positions else 0
    if len(ids) < 2:
        return np.full(n, np.nan)
    d = np.hypot(*(positions[ids[0]] - positions[ids[1]]).T)
    return d / px_per_cm if px_per_cm else d


def speeds(positions: dict[str, np.ndarray], fps: float, px_per_cm: float | None = None, smooth_seconds: float = 0.2) -> dict[str, np.ndarray]:
    out = {}
    win = max(1, int(round(smooth_seconds * fps)))
    for a, pos in positions.items():
        filled = fill_small_gaps(pos)
        step = np.vstack([[np.nan, np.nan], np.diff(filled, axis=0)])
        v = np.hypot(step[:, 0], step[:, 1]) * fps
        if px_per_cm:
            v = v / px_per_cm
        if win > 1:
            v = pd.Series(v).rolling(win, min_periods=1, center=True).mean().to_numpy()
        out[a] = v
    return out


def occupancy_heatmap(positions: dict[str, np.ndarray], frame_shape: tuple[int, int], bins: tuple[int, int] = (48, 64), sigma: float = 1.5, animals: list[str] | None = None) -> np.ndarray:
    h, w = frame_shape
    grid = np.zeros(bins, dtype=float)
    for a, pos in positions.items():
        if animals is not None and a not in animals:
            continue
        ok = ~np.isnan(pos).any(axis=1)
        if not ok.any():
            continue
        xs = np.clip((pos[ok, 0] / w * bins[1]).astype(int), 0, bins[1] - 1)
        ys = np.clip((pos[ok, 1] / h * bins[0]).astype(int), 0, bins[0] - 1)
        np.add.at(grid, (ys, xs), 1.0)
    if sigma > 0:
        grid = cv2.GaussianBlur(grid, (0, 0), sigma)
    m = grid.max()
    return grid / m if m > 0 else grid


def points_in_polygon(points: np.ndarray, polygon: list[tuple[float, float]]) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    poly = np.asarray(polygon, dtype=float)
    inside = np.zeros(len(pts), dtype=bool)
    if len(poly) < 3:
        return inside
    x, y = pts[:, 0], pts[:, 1]
    valid = ~np.isnan(x) & ~np.isnan(y)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        with np.errstate(invalid="ignore", divide="ignore"):
            crosses = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        inside ^= crosses & valid
        j = i
    return inside


def zone_occupancy(positions: dict[str, np.ndarray], zones: dict[str, list[tuple[float, float]]], fps: float) -> pd.DataFrame:
    rows = []
    for zname, poly in zones.items():
        for a, pos in positions.items():
            inside = points_in_polygon(pos, poly)
            tracked = int((~np.isnan(pos).any(axis=1)).sum())
            rows.append(
                {
                    "zone": zname,
                    "animal": a,
                    "frames": int(inside.sum()),
                    "seconds": float(inside.sum()) / (fps or 30.0),
                    "percent": 100.0 * inside.sum() / tracked if tracked else 0.0,
                }
            )
    return pd.DataFrame(rows, columns=["zone", "animal", "frames", "seconds", "percent"])


@dataclass
class Calibration:
    corners: list[tuple[float, float]] = field(default_factory=list)
    width_cm: float = 0.0
    height_cm: float | None = None

    @property
    def px_per_cm(self) -> float | None:
        if len(self.corners) != 4 or self.width_cm <= 0:
            return None
        c = np.asarray(self.corners, dtype=float)
        width_px = (np.hypot(*(c[1] - c[0])) + np.hypot(*(c[2] - c[3]))) / 2
        scales = [width_px / self.width_cm]
        if self.height_cm and self.height_cm > 0:
            height_px = (np.hypot(*(c[3] - c[0])) + np.hypot(*(c[2] - c[1]))) / 2
            scales.append(height_px / self.height_cm)
        return float(np.mean(scales))

    def to_dict(self) -> dict:
        return {"corners": [list(map(float, p)) for p in self.corners], "width_cm": self.width_cm, "height_cm": self.height_cm}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        return cls([tuple(p) for p in d.get("corners", [])], float(d.get("width_cm", 0.0)), d.get("height_cm"))
