from __future__ import annotations

import os

import cv2
import pandas as pd

_COLORS = [(66, 133, 244), (219, 68, 55), (15, 157, 88), (244, 160, 0)]


def _prepare_overlay_data(
    tidy: pd.DataFrame, predictions: pd.DataFrame | None, review_frames: pd.DataFrame | None
):
    track_ids = sorted(tidy["track_id"].unique())
    color_of = {t: _COLORS[i % len(_COLORS)] for i, t in enumerate(track_ids)}
    by_frame = {frame: g for frame, g in tidy.groupby("frame")}
    pred_by_frame = predictions.set_index("frame")["behavior"].to_dict() if predictions is not None else {}

    review_flagged = set()
    if review_frames is not None:
        for _, row in review_frames.iterrows():
            review_flagged.update(range(int(row.frame_start), int(row.frame_end) + 1))

    return color_of, by_frame, pred_by_frame, review_flagged


def _draw_overlay(frame, frame_idx, color_of, by_frame, pred_by_frame, review_flagged, width):
    for _, pt in by_frame.get(frame_idx, pd.DataFrame()).iterrows():
        color = color_of[pt["track_id"]]
        cv2.circle(frame, (int(pt["x"]), int(pt["y"])), 4, color, -1)
        cv2.putText(
            frame,
            str(pt["track_id"]),
            (int(pt["x"]) + 6, int(pt["y"]) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
        )

    if frame_idx in pred_by_frame:
        cv2.putText(
            frame,
            f"behavior: {pred_by_frame[frame_idx]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    if frame_idx in review_flagged:
        cv2.putText(frame, "REVIEW", (width - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return frame


def annotate_video(
    video_path: str,
    tidy: pd.DataFrame,
    output_path: str,
    predictions: pd.DataFrame | None = None,
    review_frames: pd.DataFrame | None = None,
) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    overlay_data = _prepare_overlay_data(tidy, predictions, review_frames)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = _draw_overlay(frame, frame_idx, *overlay_data, width=width)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return output_path


def annotate_frames(
    video_path: str,
    tidy: pd.DataFrame,
    output_dir: str,
    predictions: pd.DataFrame | None = None,
    review_frames: pd.DataFrame | None = None,
    image_ext: str = "jpg",
    every_n: int = 1,
) -> list[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    os.makedirs(output_dir, exist_ok=True)
    overlay_data = _prepare_overlay_data(tidy, predictions, review_frames)

    written = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % every_n == 0:
            frame = _draw_overlay(frame, frame_idx, *overlay_data, width=width)
            out_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.{image_ext}")
            cv2.imwrite(out_path, frame)
            written.append(out_path)
        frame_idx += 1

    cap.release()
    return written
