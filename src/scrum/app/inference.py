from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


from scrum.app.results import DetectionResult
from scrum.paths import find_model
from scrum.pipeline.behavior_classifier import predict as predict_behavior
from scrum.pipeline.blob_tracker import TrackerConfig, label_contact_bouts, track_video
from scrum.pipeline.features import compute_features as compute_keypoint_features
from scrum.pipeline.features_centroid import compute_features as compute_centroid_features
from scrum.pipeline.features_centroid import from_blob_tracks

SCRUM_MODEL_PATH = find_model("calms21_sbm_centroid_hgb.joblib")
CALMS21_MODEL_PATH = find_model("calms21_sbm_hgb.joblib")

MERGED_ID = "rat_0+rat_1"


def video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def _video_fps(video_path: str) -> float:
    return video_info(video_path)["fps"]


def _display_tracks_from_blob_results(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        for t in r["tracks"]:
            if t.get("contact") and t.get("split"):
                for part in t["split"]:
                    x, y = part["display_centroid"]
                    rows.append({"frame": r["frame"], "track_id": part["track_id"], "keypoint": "centroid", "x": x, "y": y})
                continue
            x, y = t.get("display_centroid", t["centroid"])
            rows.append({"frame": r["frame"], "track_id": t["track_id"], "keypoint": "centroid", "x": x, "y": y})
    return pd.DataFrame(rows, columns=["frame", "track_id", "keypoint", "x", "y"])


def _outline_entries_from_blob_results(results: list[dict]) -> dict[int, list[dict]]:
    outlines: dict[int, list[dict]] = {}
    for r in results:
        entries = []
        for t in r["tracks"]:
            if t.get("contours"):
                entries.append(
                    {"track_id": t["track_id"], "contours": list(t["contours"]), "merged": bool(t.get("contact")), "split": False}
                )
            for part in t.get("split") or []:
                entries.append({"track_id": part["track_id"], "contours": list(part["contours"]), "merged": True, "split": True})
        if entries:
            outlines[r["frame"]] = entries
    return outlines


def _csv_rows_from_blob_results(results: list[dict]) -> list[dict]:
    rows = []
    for r in results:
        for t in r["tracks"]:
            cx, cy = t["centroid"]
            _, _, w, h = t["bbox"]
            ellipse = t.get("ellipse")
            major, minor, angle = ellipse if ellipse is not None else (None, None, None)
            rows.append(
                {
                    "frame": r["frame"],
                    "track_id": t["track_id"],
                    "centroid_x": cx,
                    "centroid_y": cy,
                    "w": w,
                    "h": h,
                    "major": major,
                    "minor": minor,
                    "angle": angle,
                    "merged": bool(t.get("merged", t.get("contact", False))),
                    "split": t.get("split") or [],
                }
            )
    return rows


def _heuristic_predictions(results: list[dict]) -> pd.DataFrame:
    bouts = label_contact_bouts(results)
    bout_label_by_frame = {}
    for b in bouts:
        for f in range(int(b["frame_start"]), int(b["frame_end"]) + 1):
            bout_label_by_frame[f] = b["label"]
    rows = [
        {"frame": r["frame"], "behavior": bout_label_by_frame.get(r["frame"], "none"), "confidence": np.nan}
        for r in results
    ]
    return pd.DataFrame(rows, columns=["frame", "behavior", "confidence"])


def run_scrum_detection_full(
    video_path: str,
    progress=None,
    use_split: bool = True,
    model_path: str | Path | None = None,
    tracker_config: TrackerConfig | None = None,
) -> DetectionResult:
    info = video_info(video_path)
    cfg = tracker_config or TrackerConfig(split_contacts=use_split)
    results = track_video(video_path, cfg, progress=progress)
    display_tracks = _display_tracks_from_blob_results(results)
    outlines = _outline_entries_from_blob_results(results)

    n = max(info["n_frames"], len(results))
    merged = np.zeros(n, dtype=bool)
    for r in results:
        if any(t.get("contact") for t in r["tracks"]):
            merged[r["frame"]] = True

    model_path = Path(model_path) if model_path else SCRUM_MODEL_PATH
    source = "heuristic (no trained SCRUM model found)"
    predictions = None
    if model_path.exists() and results:
        try:
            centroid_tidy = from_blob_tracks(_csv_rows_from_blob_results(results), use_split=use_split)
            features = compute_centroid_features(centroid_tidy, fps=info["fps"])
            predictions = predict_behavior(features, str(model_path), smooth_min_run_frames=5, smoothing="viterbi")
            source = "SCRUM model (trained on CalMS21, centroid/bbox features)"
            if use_split:
                source += ", touching animals split"
        except ValueError as e:
            source = f"heuristic (model features could not be built: {e})"
    if predictions is None:
        predictions = _heuristic_predictions(results)

    return DetectionResult(
        video_path=video_path, fps=info["fps"], n_frames=n, width=info["width"], height=info["height"],
        source=source, mode="scrum", tracks=display_tracks, predictions=predictions, outlines=outlines, merged=merged,
    )


def run_scrum_detection(video_path: str) -> tuple[pd.DataFrame, pd.DataFrame, str, dict]:
    res = run_scrum_detection_full(video_path)
    return res.tracks, res.predictions, res.source, res.outlines


def run_calms21_keypoint_detection(tracks: pd.DataFrame, fps: float = 30.0) -> pd.DataFrame:
    if not CALMS21_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No keypoint model found at {CALMS21_MODEL_PATH}. Add one with: scrum models --add <file.joblib>"
        )
    features = compute_keypoint_features(tracks, fps=fps)
    return predict_behavior(
        features, str(CALMS21_MODEL_PATH), smooth_min_run_frames=5, smoothing="viterbi"
    )
