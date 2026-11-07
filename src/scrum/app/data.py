from __future__ import annotations

from pathlib import Path

import pandas as pd


BEHAVIOR_COLORS = {
    "none": (142, 142, 147),
    "sniffing": (10, 132, 255),
    "biting": (255, 69, 58),
    "mounting": (255, 159, 10),
}
_EXTRA_COLORS = [(48, 209, 88), (191, 90, 242), (100, 210, 255), (255, 100, 130), (172, 142, 104), (94, 92, 230)]


def behavior_rgb(name: str) -> tuple[int, int, int]:
    if name in BEHAVIOR_COLORS:
        return BEHAVIOR_COLORS[name]
    h = sum((i + 1) * ord(c) for i, c in enumerate(str(name)))
    return _EXTRA_COLORS[h % len(_EXTRA_COLORS)]


def behavior_bgr(name: str) -> tuple[int, int, int]:
    r, g, b = behavior_rgb(name)
    return (b, g, r)


def load_tracks(csv_path: str | Path) -> pd.DataFrame:
    suffix = Path(csv_path).suffix.lower()
    if suffix in (".h5", ".hdf5", ".slp"):
        from scrum.pipeline.pose_import import load_pose_file

        return load_pose_file(csv_path)[["frame", "track_id", "keypoint", "x", "y"]]
    if suffix == ".csv":
        from scrum.pipeline.pose_import import detect_format

        if detect_format(csv_path) in ("dlc", "sleap_csv"):
            from scrum.pipeline.pose_import import load_pose_file

            return load_pose_file(csv_path)[["frame", "track_id", "keypoint", "x", "y"]]
    df = pd.read_csv(csv_path)

    if "keypoint" in df.columns:
        return df[["frame", "track_id", "keypoint", "x", "y"]].copy()

    if "display_x" in df.columns and "display_y" in df.columns:
        out = df.copy()
        out["x"] = out["display_x"].fillna(out.get("centroid_x"))
        out["y"] = out["display_y"].fillna(out.get("centroid_y"))
        out["keypoint"] = "centroid"
        return out[["frame", "track_id", "keypoint", "x", "y"]]

    if "x" in df.columns and "y" in df.columns:
        out = df.copy()
        out["keypoint"] = out.get("keypoint", "point")
        return out[["frame", "track_id", "keypoint", "x", "y"]]

    raise ValueError(f"Don't recognize the column layout in {csv_path}: {list(df.columns)}")


def load_predictions(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    behavior_col = "behavior" if "behavior" in df.columns else "behavior_pred"
    if behavior_col not in df.columns:
        raise ValueError(f"No behavior column found in {csv_path}: {list(df.columns)}")
    out = df[["frame", behavior_col]].rename(columns={behavior_col: "behavior"})
    if "confidence" in df.columns:
        out["confidence"] = df["confidence"]
    return out


def csv_kind(csv_path: str | Path) -> str:
    cols = set(pd.read_csv(csv_path, nrows=1).columns)
    return "predictions" if cols & {"behavior", "behavior_pred"} and "x" not in cols else "tracks"


def segments_from_predictions(predictions: pd.DataFrame) -> list[dict]:
    if predictions.empty:
        return []
    predictions = predictions.sort_values("frame").reset_index(drop=True)
    segments = []
    start_frame = int(predictions.loc[0, "frame"])
    current = predictions.loc[0, "behavior"]
    prev_frame = start_frame
    for _, row in predictions.iloc[1:].iterrows():
        f, behavior = int(row["frame"]), row["behavior"]
        if behavior != current:
            segments.append({"start": start_frame, "end": prev_frame, "behavior": current})
            start_frame = f
            current = behavior
        prev_frame = f
    segments.append({"start": start_frame, "end": prev_frame, "behavior": current})
    return segments


def tracks_by_frame(tracks: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {int(frame): g for frame, g in tracks.groupby("frame")}
