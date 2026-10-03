from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.calms21_import import sequence_to_tidy, sequence_to_labels, KEYPOINT_NAMES, TRACK_IDS
from pipeline.features import compute_features
from pipeline.behavior_classifier import predict

_MOUSE_COLORS = {"resident": (66, 133, 244), "intruder": (219, 68, 55)}
_SKELETON = [
    ("nose", "neck"),
    ("left_ear", "neck"),
    ("right_ear", "neck"),
    ("neck", "left_hip"),
    ("neck", "right_hip"),
    ("left_hip", "tailbase"),
    ("right_hip", "tailbase"),
]
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BG = (0, 0, 0)
_MATCH_COLOR = (0, 200, 0)
_MISMATCH_COLOR = (0, 140, 255)


def _draw_skeleton(frame, kp_frame, track_id):
    color = _MOUSE_COLORS[track_id]
    pts = {name: (int(kp_frame[name][0]), int(kp_frame[name][1])) for name in KEYPOINT_NAMES}
    for a, b in _SKELETON:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for name, pt in pts.items():
        cv2.circle(frame, pt, 4, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 4, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_banner(frame, pred, true):
    text = f"predicted: {pred}   true: {true}"
    color = _MATCH_COLOR if pred == true else _MISMATCH_COLOR
    (tw, th), baseline = cv2.getTextSize(text, _FONT, 0.7, 2)
    cx = frame.shape[1] // 2 - tw // 2
    cv2.rectangle(frame, (cx - 10, 8), (cx + tw + 10, 8 + th + baseline + 10), _BG, -1)
    cv2.putText(frame, text, (cx, 8 + th + 5), _FONT, 0.7, color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument(
        "--sequence-json", required=True, help="one-sequence json, e.g. {'task1/train/...': {...}}"
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--smooth-min-run-frames", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.sequence_json) as f:
        data = json.load(f)
    name, seq = next(iter(data.items()))
    print(f"sequence: {name}")

    tidy = sequence_to_tidy(seq)
    true_labels = sequence_to_labels(seq, as_sbm=True)
    feats = compute_features(tidy, fps=args.fps)
    preds = predict(feats, args.model, smooth_min_run_frames=args.smooth_min_run_frames)
    merged = preds.merge(true_labels, on="frame", suffixes=("_pred", "_true")).set_index("frame")

    kp_lookup = {}
    for (frame, track_id), rows in tidy.groupby(["frame", "track_id"]):
        kp_lookup[(frame, track_id)] = {r.keypoint: (r.x, r.y) for r in rows.itertuples()}

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_idx = 0
    n_match = 0
    n_scored = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for track_id in TRACK_IDS:
            kp_frame = kp_lookup.get((frame_idx, track_id))
            if kp_frame:
                _draw_skeleton(frame, kp_frame, track_id)
        if frame_idx in merged.index:
            row = merged.loc[frame_idx]
            _draw_banner(frame, row["behavior_pred"], row["behavior_true"])
            n_scored += 1
            n_match += int(row["behavior_pred"] == row["behavior_true"])
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    merged.reset_index().to_csv(out_dir / "predictions.csv", index=False)
    print(f"{frame_idx} frames rendered -> {out_dir / 'annotated.mp4'}")
    print(
        f"frame-level agreement with CalMS21's own ground truth: {n_match}/{n_scored} ({100 * n_match / n_scored:.1f}%)"
    )
    print(f"predictions saved -> {out_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
