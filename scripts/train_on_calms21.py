from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.calms21_import import load_calms21_split, iter_sequences, sequence_to_tidy, sequence_to_labels
from pipeline.features import compute_features
from pipeline.behavior_classifier import DEFAULT_BEHAVIOR_CLASSES, train, predict

FRAME_OFFSET = 10_000_000


def _build_features_and_labels(json_path: str, fps: float, limit: int | None):
    print(f"loading {json_path} (large file, can take ~10-20s and real memory)...")
    data = load_calms21_split(json_path)

    all_features, all_labels = [], []
    seq_idx = 0
    for name, seq in iter_sequences(data):
        if limit and seq_idx >= limit:
            break
        try:
            tidy = sequence_to_tidy(seq)
            labels = sequence_to_labels(seq, as_sbm=True)
            feats = compute_features(tidy, fps=fps)

            feats["frame"] = feats["frame"] + seq_idx * FRAME_OFFSET
            labels["frame"] = labels["frame"] + seq_idx * FRAME_OFFSET

            all_features.append(feats)
            all_labels.append(labels)
            print(f"  {name}: {len(feats)} frames, labels: {labels['behavior'].value_counts().to_dict()}")
        except Exception as e:
            print(f"  {name}: skipped ({e})")
        seq_idx += 1

    if not all_features:
        return None, None
    return pd.concat(all_features, ignore_index=True), pd.concat(all_labels, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-json", required=True, help="path to calms21_task1_train.json")
    ap.add_argument(
        "--test-json",
        default=None,
        help="path to calms21_task1_test.json (optional, for a true held-out check)",
    )
    ap.add_argument("--model-out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument(
        "--limit", type=int, default=None, help="only use the first N sequences per split (smoke test)"
    )
    ap.add_argument("--model", choices=["hgb", "rf"], default="hgb")
    args = ap.parse_args()

    features, labels = _build_features_and_labels(args.train_json, args.fps, args.limit)
    if features is None:
        print("Nothing converted successfully from the train file, nothing to train on.")
        return

    print(f"\nCombined train: {len(features)} frames")
    print("Overall label distribution (train):")
    print(labels["behavior"].value_counts())

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    result = train(
        features, labels, args.model_out, behavior_classes=DEFAULT_BEHAVIOR_CLASSES, model=args.model
    )
    print(f"\nSaved model to {args.model_out} ({result['n_train']} train / {result['n_test']} test frames)")
    print("80/20 holdout of the train file (same split style as train_on_blossom.py):")
    print(f"macro F1: {result['macro_f1']:.3f}  weighted F1: {result['weighted_f1']:.3f}\n")
    print(result["report"])

    if args.test_json:
        test_features, test_labels = _build_features_and_labels(args.test_json, args.fps, args.limit)
        if test_features is None:
            print("Nothing converted successfully from the test file; skipping the held-out check.")
            return
        print(
            f"\nCombined test (CalMS21's own held-out split, never seen during training): {len(test_features)} frames"
        )
        preds = predict(test_features, args.model_out, smooth_min_run_frames=0)
        merged = preds.merge(test_labels, on="frame", suffixes=("_pred", "_true"))
        if merged.empty:
            print("No overlapping frames between test features and test labels - check the 'frame' columns.")
            return
        from sklearn.metrics import classification_report, f1_score

        macro_f1 = f1_score(
            merged["behavior_true"], merged["behavior_pred"], average="macro", zero_division=0
        )
        weighted_f1 = f1_score(
            merged["behavior_true"], merged["behavior_pred"], average="weighted", zero_division=0
        )
        print("CalMS21's own held-out test.json, never seen during training:")
        print(f"macro F1: {macro_f1:.3f}  weighted F1: {weighted_f1:.3f}\n")
        print(classification_report(merged["behavior_true"], merged["behavior_pred"], zero_division=0))


if __name__ == "__main__":
    main()
