from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .io_utils import centroids


@dataclass
class ResolutionConfig:
    contact_radius_px: float = 40.0
    score_threshold: float = 0.4
    min_clean_frames: int = 5
    max_gap_to_bridge: int = 90


def _clean_frame_mask(
    cents: pd.DataFrame, mean_score: pd.Series, cfg: ResolutionConfig, n_tracks: int
) -> pd.Series:
    counts = cents.groupby("frame")["track_id"].nunique()
    full_occupancy = counts.reindex(cents["frame"].unique(), fill_value=0) >= n_tracks

    min_dist = _min_pairwise_distance_per_frame(cents)
    well_separated = min_dist > cfg.contact_radius_px

    confident = mean_score >= cfg.score_threshold

    frames = sorted(cents["frame"].unique())
    idx = pd.Index(frames, name="frame")
    clean = (
        full_occupancy.reindex(idx, fill_value=False)
        & well_separated.reindex(idx, fill_value=False)
        & confident.reindex(idx, fill_value=False)
    )
    return clean


def _min_pairwise_distance_per_frame(cents: pd.DataFrame) -> pd.Series:
    out = {}
    for frame, group in cents.groupby("frame"):
        pts = group[["cx", "cy"]].to_numpy()
        if len(pts) < 2:
            out[frame] = np.inf
            continue
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        out[frame] = float(d.min())
    return pd.Series(out)


def _find_segments(occluded: pd.Series) -> list[tuple[int, int]]:
    frames = occluded.index.to_numpy()
    mask = occluded.to_numpy()
    segments = []
    start = None
    for i, (frame, bad) in enumerate(zip(frames, mask)):
        if bad and start is None:
            start = frame
        if not bad and start is not None:
            segments.append((start, frames[i - 1]))
            start = None
    if start is not None:
        segments.append((start, frames[-1]))
    return segments


def _predict_positions(track_history: pd.DataFrame, target_frame: int) -> dict[str, np.ndarray]:
    preds = {}
    for track_id, g in track_history.groupby("track_id"):
        g = g.sort_values("frame")
        if len(g) < 2:
            preds[track_id] = g[["cx", "cy"]].to_numpy()[-1]
            continue
        p0, p1 = g[["cx", "cy"]].to_numpy()[[0, -1]]
        f0, f1 = g["frame"].to_numpy()[[0, -1]]
        velocity = (p1 - p0) / max(f1 - f0, 1)
        preds[track_id] = p1 + velocity * (target_frame - f1)
    return preds


def resolve_identities(tidy: pd.DataFrame, cfg: ResolutionConfig | None = None):
    cfg = cfg or ResolutionConfig()
    tidy = tidy.copy()
    cents = centroids(tidy)
    n_tracks = tidy["track_id"].nunique()

    mean_score = tidy.groupby("frame")["score"].mean()
    clean = _clean_frame_mask(cents, mean_score, cfg, n_tracks)
    segments = _find_segments(~clean)

    stable_map: dict[str, str] = {}
    first_clean_frame = clean[clean].index.min() if clean.any() else cents["frame"].min()
    for tid in cents.loc[cents["frame"] == first_clean_frame, "track_id"]:
        stable_map[tid] = tid

    corrected_col = tidy["track_id"].astype(str)
    review_rows = []
    all_frames = sorted(cents["frame"].unique())

    clean_frames_sorted = np.array(sorted(f for f in all_frames if clean.get(f, False)))

    for seg_start, seg_end in segments:
        n_gap = seg_end - seg_start + 1
        idx_before = np.searchsorted(clean_frames_sorted, seg_start, side="left")
        before = clean_frames_sorted[max(0, idx_before - cfg.min_clean_frames) : idx_before].tolist()
        idx_after = np.searchsorted(clean_frames_sorted, seg_end, side="right")
        after = clean_frames_sorted[idx_after : idx_after + cfg.min_clean_frames].tolist()

        if n_gap > cfg.max_gap_to_bridge or len(before) < 2 or len(after) < 1:
            review_rows.append(
                {
                    "frame_start": seg_start,
                    "frame_end": seg_end,
                    "n_frames": n_gap,
                    "risk": "high",
                    "auto_corrected": False,
                }
            )
            continue

        history = cents[cents["frame"].isin(before)].copy()
        history["track_id"] = history["track_id"].map(lambda t: stable_map.get(t, t))
        target_frame = after[0]
        predicted = _predict_positions(history, target_frame)

        actual = cents[cents["frame"] == target_frame]
        stable_ids = list(predicted.keys())
        raw_ids = actual["track_id"].tolist()
        if len(stable_ids) != len(raw_ids) or not raw_ids:
            review_rows.append(
                {
                    "frame_start": seg_start,
                    "frame_end": seg_end,
                    "n_frames": n_gap,
                    "risk": "high",
                    "auto_corrected": False,
                }
            )
            continue

        cost = np.zeros((len(stable_ids), len(raw_ids)))
        for i, sid in enumerate(stable_ids):
            for j, rid in enumerate(raw_ids):
                actual_pos = actual.loc[actual["track_id"] == rid, ["cx", "cy"]].to_numpy()[0]
                cost[i, j] = np.linalg.norm(predicted[sid] - actual_pos)

        row_ind, col_ind = linear_sum_assignment(cost)
        new_map = {raw_ids[j]: stable_ids[i] for i, j in zip(row_ind, col_ind)}

        swap_detected = any(raw != stable for raw, stable in new_map.items() if raw in stable_map.values())
        stable_map.update(new_map)

        next_seg_start = (
            segments[segments.index((seg_start, seg_end)) + 1][0]
            if (seg_start, seg_end) != segments[-1]
            else None
        )
        span_mask = (tidy["frame"] >= target_frame) & (
            (tidy["frame"] < next_seg_start) if next_seg_start is not None else True
        )
        corrected_col.loc[span_mask] = tidy.loc[span_mask, "track_id"].map(lambda t: stable_map.get(t, t))

        review_rows.append(
            {
                "frame_start": seg_start,
                "frame_end": seg_end,
                "n_frames": n_gap,
                "risk": "low" if not swap_detected else "medium",
                "auto_corrected": True,
            }
        )

    resolved = tidy.copy()
    resolved["track_id"] = corrected_col
    review_frames = pd.DataFrame(
        review_rows, columns=["frame_start", "frame_end", "n_frames", "risk", "auto_corrected"]
    )
    return resolved, review_frames
