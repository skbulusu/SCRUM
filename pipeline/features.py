from __future__ import annotations

import numpy as np
import pandas as pd

KEYPOINTS = {"head": "nose", "tail": "tailbase"}


def _pivot_point(tidy: pd.DataFrame, keypoint: str) -> pd.DataFrame:
    df = tidy[tidy["keypoint"] == keypoint]
    return df.pivot_table(index="frame", columns="track_id", values=["x", "y"])


def compute_features(
    tidy: pd.DataFrame, fps: float = 30.0, windows: tuple[int, ...] = (5, 15)
) -> pd.DataFrame:
    track_ids = sorted(tidy["track_id"].unique())
    if len(track_ids) != 2:
        raise ValueError(f"compute_features expects exactly 2 tracks, got {len(track_ids)}: {track_ids}")
    a, b = track_ids

    head = _pivot_point(tidy, KEYPOINTS["head"])
    tail = _pivot_point(tidy, KEYPOINTS["tail"])
    frames = sorted(set(head.index) | set(tail.index))
    head = head.reindex(frames)
    tail = tail.reindex(frames)

    def pt(df, coord, track):
        return df[coord][track] if (coord, track) in df.columns else pd.Series(np.nan, index=df.index)

    centroid = {
        t: pd.DataFrame(
            {"x": (pt(head, "x", t) + pt(tail, "x", t)) / 2, "y": (pt(head, "y", t) + pt(tail, "y", t)) / 2}
        )
        for t in (a, b)
    }

    out = pd.DataFrame(index=frames)
    out.index.name = "frame"

    out["centroid_dist"] = np.hypot(centroid[a]["x"] - centroid[b]["x"], centroid[a]["y"] - centroid[b]["y"])
    out["nose_to_nose_dist"] = np.hypot(
        pt(head, "x", a) - pt(head, "x", b), pt(head, "y", a) - pt(head, "y", b)
    )
    out["A_nose_to_B_tail_dist"] = np.hypot(
        pt(head, "x", a) - pt(tail, "x", b), pt(head, "y", a) - pt(tail, "y", b)
    )
    out["B_nose_to_A_tail_dist"] = np.hypot(
        pt(head, "x", b) - pt(tail, "x", a), pt(head, "y", b) - pt(tail, "y", a)
    )

    body_len = {
        t: np.hypot(pt(head, "x", t) - pt(tail, "x", t), pt(head, "y", t) - pt(tail, "y", t)) for t in (a, b)
    }
    smaller_len = np.minimum(body_len[a], body_len[b])
    out["bbox_overlap_frac"] = np.clip(
        1.0 - out["centroid_dist"] / smaller_len.replace(0, np.nan), 0.0, 1.0
    ).fillna(0.0)

    for t in (a, b):
        dx = centroid[t]["x"].diff()
        dy = centroid[t]["y"].diff()
        out[f"{t}_speed"] = np.hypot(dx, dy) * fps
        out[f"{t}_heading"] = np.arctan2(
            pt(head, "y", t) - pt(tail, "y", t), pt(head, "x", t) - pt(tail, "x", t)
        )

    heading_diff = out[f"{a}_heading"] - out[f"{b}_heading"]
    out["relative_heading"] = np.abs(np.arctan2(np.sin(heading_diff), np.cos(heading_diff)))

    out = out.rename(
        columns={
            f"{a}_speed": "A_speed",
            f"{b}_speed": "B_speed",
            f"{a}_heading": "A_heading",
            f"{b}_heading": "B_heading",
        }
    )

    roll_cols = [c for c in out.columns if c not in ("A_heading", "B_heading")]
    for w in windows:
        rolled = out[roll_cols].rolling(window=w, min_periods=1).mean()
        rolled.columns = [f"{c}_roll{w}" for c in roll_cols]
        out = pd.concat([out, rolled], axis=1)

    out.attrs["track_a"] = a
    out.attrs["track_b"] = b
    return out.reset_index()
