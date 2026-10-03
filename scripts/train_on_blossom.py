from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.blossom_import import EXPERIMENT_CLASSES, SBM_CAVEATS, available_video_ids, convert_video
from pipeline.io_utils import load_tracking
from pipeline.identity_resolution import ResolutionConfig, resolve_identities
from pipeline.features import compute_features
from pipeline.behavior_classifier import train

FRAME_OFFSET = 1_000_000


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dir", required=True, help="path to the cloned Mouse-Behavior-Classifier-Train repo"
    )
    ap.add_argument(
        "--experiment", choices=["behavior", "aggression", "sniffing_biting_mounting"], default="behavior"
    )
    ap.add_argument("--model-out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--contact-radius-px", type=float, default=40.0)
    ap.add_argument("--limit", type=int, default=None, help="only use the first N videos (smoke test)")
    ap.add_argument("--model", choices=["hgb", "rf"], default="hgb")
    ap.add_argument(
        "--cache-dir", default=None, help="where converted .h5/.csv files go (default: a temp dir)"
    )
    args = ap.parse_args()

    if args.experiment == "sniffing_biting_mounting":
        print(
            "this data has no real 'mounting' labels, it's an aggression paradigm, not a mating study."
        )
        for cls, note in SBM_CAVEATS.items():
            print(f"  {cls}: {note}")
        print()

    video_ids = available_video_ids(args.data_dir)
    if not video_ids:
        print(f"No videos with both a DLC csv and an annotation file found under {args.data_dir}")
        return
    if args.limit:
        video_ids = video_ids[: args.limit]
    print(f"Found {len(video_ids)} usable video(s): {video_ids}")

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(tempfile.mkdtemp(prefix="blossom_convert_"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Converted files going to: {cache_dir}")

    all_features, all_labels = [], []
    for vid in video_ids:
        try:
            h5_path, labels_path = convert_video(args.data_dir, vid, cache_dir, experiment=args.experiment)
            tidy = load_tracking(str(h5_path))
            resolved, _review = resolve_identities(
                tidy, ResolutionConfig(contact_radius_px=args.contact_radius_px)
            )
            feats = compute_features(resolved, fps=args.fps)
            labels = pd.read_csv(labels_path)

            feats["frame"] = feats["frame"] + vid * FRAME_OFFSET
            labels["frame"] = labels["frame"] + vid * FRAME_OFFSET

            all_features.append(feats)
            all_labels.append(labels)
            print(
                f"  video {vid}: {len(feats)} frames, labels: {labels['behavior'].value_counts().to_dict()}"
            )
        except Exception as e:
            print(f"  video {vid}: skipped ({e})")

    if not all_features:
        print("Nothing converted successfully, nothing to train on.")
        return

    features = pd.concat(all_features, ignore_index=True)
    labels = pd.concat(all_labels, ignore_index=True)
    print(f"\nCombined: {len(features)} frames across {len(video_ids)} videos")
    print("Overall label distribution:")
    print(labels["behavior"].value_counts())

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    result = train(
        features,
        labels,
        args.model_out,
        behavior_classes=EXPERIMENT_CLASSES[args.experiment],
        model=args.model,
    )
    print(f"\nSaved model to {args.model_out} ({result['n_train']} train / {result['n_test']} test frames)")
    print(f"macro F1: {result['macro_f1']:.3f}  weighted F1: {result['weighted_f1']:.3f}\n")
    print(result["report"])


if __name__ == "__main__":
    main()
