from __future__ import annotations

import numpy as np
import pandas as pd


def compute_features(
    tidy: pd.DataFrame, fps: float = 25.0, windows: tuple[int, ...] = (5, 15)
) -> pd.DataFrame:
    track_ids = sorted(tidy["track_id"].unique())
    if len(track_ids) != 2:
        raise ValueError(f"compute_features expects exactly 2 tracks, got {len(track_ids)}: {track_ids}")
    a, b = track_ids
    has_bbox = "w" in tidy.columns and "h" in tidy.columns

    cols = ["cx", "cy"] + (["w", "h"] if has_bbox else [])
    wide = tidy.pivot_table(index="frame", columns="track_id", values=cols)
    frames = sorted(tidy["frame"].unique())
    wide = wide.reindex(frames)

    def get(col, t):
        return wide[col][t] if (col, t) in wide.columns else pd.Series(np.nan, index=wide.index)

    out = pd.DataFrame(index=frames)
    out.index.name = "frame"

    dx = get("cx", a) - get("cx", b)
    dy = get("cy", a) - get("cy", b)
    out["centroid_dist"] = np.hypot(dx, dy)

    out["closing_speed"] = -out["centroid_dist"].diff() * fps

    for t in (a, b):
        vx = get("cx", t).diff()
        vy = get("cy", t).diff()
        speed = np.hypot(vx, vy) * fps
        out[f"{t}_speed"] = speed
        out[f"{t}_accel"] = speed.diff() * fps

    if has_bbox:
        area = {t: get("w", t) * get("h", t) for t in (a, b)}

        body_scale = {t: np.sqrt(area[t].replace(0, np.nan)) for t in (a, b)}
        smaller_scale = np.minimum(body_scale[a], body_scale[b])
        out["bbox_overlap_frac"] = np.clip(1.0 - out["centroid_dist"] / smaller_scale, 0.0, 1.0).fillna(0.0)

    out = out.rename(
        columns={
            f"{a}_speed": "A_speed",
            f"{b}_speed": "B_speed",
            f"{a}_accel": "A_accel",
            f"{b}_accel": "B_accel",
        }
    )

    roll_cols = list(out.columns)
    for w in windows:
        rolled = out[roll_cols].rolling(window=w, min_periods=1).mean()
        rolled.columns = [f"{c}_roll{w}" for c in roll_cols]
        out = pd.concat([out, rolled], axis=1)

    out.attrs["track_a"] = a
    out.attrs["track_b"] = b
    return out.reset_index()


def from_blob_tracks(tracks_csv_rows: list[dict]) -> pd.DataFrame:
    rows = []
    for r in tracks_csv_rows:
        if r["track_id"] == "rat_0+rat_1":
            continue
        rows.append(
            {
                "frame": r["frame"],
                "track_id": r["track_id"],
                "cx": r["centroid_x"],
                "cy": r["centroid_y"],
                "w": r["w"],
                "h": r["h"],
            }
        )
    return pd.DataFrame(rows)
