from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.identity_resolution import ResolutionConfig, resolve_identities
from pipeline.features import compute_features
from pipeline.behavior_classifier import train, predict
from pipeline.smoothing import majority_vote_smooth, enforce_min_bout
from pipeline.evaluate import frame_level_report, event_level_report

N_FRAMES = 200
FPS = 30.0
BODY_HALF = 15.0
SWAP_FRAME = 86
SNIFF_WINDOW = (40, 55)
MOUNT_WINDOW = (150, 165)


def true_center_base(stable_id: str, t: int) -> np.ndarray:
    if stable_id == "rat_0":
        return np.array([100.0 + 3.0 * t, 300.0])
    return np.array([600.0 - 3.0 * t, 300.0])


def true_center(stable_id: str, t: int) -> np.ndarray:
    if stable_id == "rat_0":
        if SNIFF_WINDOW[0] <= t <= SNIFF_WINDOW[1]:
            return true_center_base("rat_1", t) + np.array([20.0, 0.0])
        if MOUNT_WINDOW[0] <= t <= MOUNT_WINDOW[1]:
            return true_center_base("rat_1", t)
    return true_center_base(stable_id, t)


def build_synthetic_dataset():
    rows = []
    truth_rows = []
    for t in range(N_FRAMES):
        c0, c1 = true_center("rat_0", t), true_center("rat_1", t)
        dist = float(np.linalg.norm(c0 - c1))
        score = 0.9 if dist >= 15 else 0.3

        points = {
            "rat_0": {"nose": c0 + [BODY_HALF, 0], "tailbase": c0 - [BODY_HALF, 0]},
            "rat_1": {"nose": c1 - [BODY_HALF, 0], "tailbase": c1 + [BODY_HALF, 0]},
        }

        raw_of = {"rat_0": "track_0", "rat_1": "track_1"}
        if t >= SWAP_FRAME:
            raw_of = {"rat_0": "track_1", "rat_1": "track_0"}

        for stable_id, kp in points.items():
            raw_id = raw_of[stable_id]
            for name, (x, y) in kp.items():
                rows.append(
                    {"frame": t, "track_id": raw_id, "keypoint": name, "x": x, "y": y, "score": score}
                )

        if SNIFF_WINDOW[0] <= t <= SNIFF_WINDOW[1]:
            behavior = "sniffing"
        elif MOUNT_WINDOW[0] <= t <= MOUNT_WINDOW[1]:
            behavior = "mounting"
        else:
            behavior = "none"
        truth_rows.append({"frame": t, "behavior": behavior, "true_rat_0_x": c0[0], "true_rat_1_x": c1[0]})

    return pd.DataFrame(rows), pd.DataFrame(truth_rows)


def main():
    print("building synthetic raw tracking data (simulates a SLEAP/DLC export)")
    tidy, truth = build_synthetic_dataset()
    print(
        f"{len(tidy)} keypoint rows across {tidy['frame'].nunique()} frames, raw track_ids: {sorted(tidy.track_id.unique())}"
    )

    print("\nresolving identity (occlusion / same-color swap repair)")
    resolved, review_frames = resolve_identities(
        tidy, ResolutionConfig(contact_radius_px=15, score_threshold=0.4)
    )
    print(review_frames.to_string(index=False))

    check_frame = 190
    row = resolved[(resolved.frame == check_frame) & (resolved.keypoint == "nose")]
    true_x = truth.loc[truth.frame == check_frame, "true_rat_0_x"].item()
    rat0_row = row[row.track_id == "track_0"]
    if not rat0_row.empty:
        got_x = rat0_row.x.item() - BODY_HALF
        match = abs(got_x - true_x) < 5
        print(
            f"\nsanity check @ frame {check_frame}: corrected track_0 x={got_x:.1f}, true rat_0 x={true_x:.1f} "
            f"(match: {match})"
        )

    print("\ncomputing features")
    features = compute_features(resolved, fps=FPS)
    print(features.describe().loc[["mean", "std", "min", "max"]].round(1).to_string())

    print("\ntraining the classifier")
    labels = truth[["frame", "behavior"]]
    model_path = str(Path(__file__).resolve().parent / "_demo_model.joblib")
    result = train(features, labels, model_path)
    print(f"trained on {result['n_train']} frames, held out {result['n_test']}")
    print(result["report"])

    print("\npredicting and smoothing over the full sequence")
    preds = predict(features, model_path)
    preds = majority_vote_smooth(preds, window=5)
    preds = enforce_min_bout(preds, min_frames=3)

    print("\nevaluating against synthetic ground truth")
    frame_report = frame_level_report(preds, labels)
    print(frame_report["report"])

    event_report = event_level_report(preds, labels)
    print("event-level (bout) scoring:")
    print(event_report.to_string(index=False))

    print("\ndemo complete, every pipeline stage ran without error")


if __name__ == "__main__":
    main()
