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
    has_ellipse = all(c in tidy.columns for c in ("major", "minor", "angle"))

    cols = ["cx", "cy"]
    if has_bbox:
        cols += ["w", "h"]
    if has_ellipse:
        cols += ["major", "minor", "angle"]
    wide = tidy.pivot_table(index="frame", columns="track_id", values=cols)
    frames = sorted(tidy["frame"].unique())
    wide = wide.reindex(frames)

    def get(col, t):
        return wide[col][t] if (col, t) in wide.columns else pd.Series(np.nan, index=wide.index)

    out = pd.DataFrame(index=frames)
    out.index.name = "frame"


    if "merged" in tidy.columns:
        merged_by_frame = tidy.groupby("frame")["merged"].max().astype(float)
        out["merged_blob"] = merged_by_frame.reindex(frames).fillna(0.0).to_numpy()

    dx = get("cx", a) - get("cx", b)
    dy = get("cy", a) - get("cy", b)
    out["centroid_dist"] = np.hypot(dx, dy)

    out["closing_speed"] = -out["centroid_dist"].diff() * fps

    velocity = {}
    for t in (a, b):
        vx = get("cx", t).diff()
        vy = get("cy", t).diff()
        velocity[t] = (vx, vy)
        speed = np.hypot(vx, vy) * fps
        out[f"{t}_speed"] = speed
        out[f"{t}_accel"] = speed.diff() * fps


        out[f"{t}_motion_heading"] = np.arctan2(vy, vx)


    angle_to_other = {a: np.arctan2(-dy, -dx), b: np.arctan2(dy, dx)}
    for t in (a, b):
        out[f"{t}_facing_other"] = np.cos(out[f"{t}_motion_heading"] - angle_to_other[t])

    body_scale = None
    if has_bbox:
        area = {t: get("w", t) * get("h", t) for t in (a, b)}
        bbox_scale = {t: np.sqrt(area[t].replace(0, np.nan)) for t in (a, b)}
        smaller_scale = np.minimum(bbox_scale[a], bbox_scale[b])
        out["bbox_overlap_frac"] = np.clip(1.0 - out["centroid_dist"] / smaller_scale, 0.0, 1.0).fillna(0.0)
        body_scale = (bbox_scale[a] + bbox_scale[b]) / 2

    if has_ellipse:
        orientation = {}
        for t in (a, b):
            major = get("major", t)
            minor = get("minor", t).replace(0, np.nan)
            out[f"{t}_elongation"] = (major / minor).fillna(1.0).clip(upper=10.0)
            orientation[t] = np.radians(get("angle", t))
            out[f"{t}_orientation"] = orientation[t]


        diff = np.mod(orientation[a] - orientation[b], np.pi)
        out["relative_orientation"] = np.minimum(diff, np.pi - diff)


        ellipse_scale = (get("major", a).replace(0, np.nan) + get("major", b).replace(0, np.nan)) / 2
        body_scale = ellipse_scale if body_scale is None else ellipse_scale.combine_first(body_scale)

    if body_scale is not None:
        out["centroid_dist_norm"] = (out["centroid_dist"] / body_scale).fillna(0.0)
        out["closing_speed_norm"] = (out["closing_speed"] / body_scale).fillna(0.0)
        for t in (a, b):
            out[f"{t}_speed_norm"] = (out[f"{t}_speed"] / body_scale).fillna(0.0)

    out = out.rename(
        columns={
            f"{a}_speed": "A_speed",
            f"{b}_speed": "B_speed",
            f"{a}_accel": "A_accel",
            f"{b}_accel": "B_accel",
            f"{a}_motion_heading": "A_motion_heading",
            f"{b}_motion_heading": "B_motion_heading",
            f"{a}_facing_other": "A_facing_other",
            f"{b}_facing_other": "B_facing_other",
            f"{a}_elongation": "A_elongation",
            f"{b}_elongation": "B_elongation",
            f"{a}_orientation": "A_orientation",
            f"{b}_orientation": "B_orientation",
            f"{a}_speed_norm": "A_speed_norm",
            f"{b}_speed_norm": "B_speed_norm",
        }
    )


    angle_cols = {"A_motion_heading", "B_motion_heading", "A_orientation", "B_orientation"}
    roll_cols = [c for c in out.columns if c not in angle_cols]
    for w in windows:
        rolled = out[roll_cols].rolling(window=w, min_periods=1).mean()
        rolled.columns = [f"{c}_roll{w}" for c in roll_cols]
        out = pd.concat([out, rolled], axis=1)

    out.attrs["track_a"] = a
    out.attrs["track_b"] = b
    return out.reset_index()


STABLE_IDS = ["rat_0", "rat_1"]


def from_blob_tracks(
    tracks_csv_rows: list[dict], use_split: bool = True, min_split_confidence: float = 0.3
) -> pd.DataFrame:
    by_frame: dict[int, list[dict]] = {}
    for r in tracks_csv_rows:
        by_frame.setdefault(r["frame"], []).append(r)

    def _row(frame, tid, cx, cy, w, h, major, minor, angle, merged):
        return {
            "frame": frame,
            "track_id": tid,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "major": major,
            "minor": minor,
            "angle": angle,
            "merged": merged,
        }

    rows = []
    for frame, frame_rows in by_frame.items():
        merged = next((r for r in frame_rows if r["track_id"] == "rat_0+rat_1"), None)
        if merged is not None:
            split = {p["track_id"]: p for p in (merged.get("split") or [])} if use_split else {}
            if any(p.get("confidence", 1.0) < min_split_confidence for p in split.values()):
                split = {}
            for tid in STABLE_IDS:
                part = split.get(tid)
                if part is not None and len(split) == 2:
                    ell = part.get("ellipse") or (None, None, None)
                    rows.append(
                        _row(
                            frame, tid, part["centroid"][0], part["centroid"][1],
                            part["bbox"][2], part["bbox"][3], ell[0], ell[1], ell[2], True,
                        )
                    )
                else:
                    rows.append(
                        _row(
                            frame, tid, merged["centroid_x"], merged["centroid_y"], merged["w"],
                            merged["h"], merged.get("major"), merged.get("minor"), merged.get("angle"), True,
                        )
                    )
            continue
        for r in frame_rows:
            if r["track_id"] not in STABLE_IDS:
                continue
            rows.append(
                _row(
                    r["frame"], r["track_id"], r["centroid_x"], r["centroid_y"], r["w"], r["h"],
                    r.get("major"), r.get("minor"), r.get("angle"), False,
                )
            )
    return pd.DataFrame(rows)
