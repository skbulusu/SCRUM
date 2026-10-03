from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.io_utils import load_tracking
from pipeline.identity_resolution import ResolutionConfig, resolve_identities
from pipeline.features import compute_features
from pipeline.behavior_classifier import predict
from pipeline.smoothing import majority_vote_smooth, enforce_min_bout
from pipeline.evaluate import frame_level_report, event_level_report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracking", required=True)
    ap.add_argument("--model", required=True, help="trained model from train_classifier.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--video", default=None, help="source video, to render an annotated overlay for QA")
    ap.add_argument(
        "--ground-truth", default=None, help="CSV with frame,behavior, if given, scores predictions"
    )
    ap.add_argument(
        "--export-frames",
        action="store_true",
        help="also write one annotated image per frame (needs --video)",
    )
    ap.add_argument(
        "--frame-every-n",
        type=int,
        default=1,
        help="with --export-frames, write only every Nth frame (default: every frame)",
    )
    ap.add_argument(
        "--frame-ext", default="jpg", choices=["jpg", "png"], help="image format for --export-frames"
    )
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--contact-radius-px", type=float, default=40.0)
    ap.add_argument("--smooth-window", type=int, default=5)
    ap.add_argument("--min-bout-frames", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tidy = load_tracking(args.tracking)
    resolved, review = resolve_identities(tidy, ResolutionConfig(contact_radius_px=args.contact_radius_px))
    review.to_csv(out_dir / "review_frames.csv", index=False)
    print(f"{len(review)} occlusion segment(s) flagged -> {out_dir / 'review_frames.csv'}")

    features = compute_features(resolved, fps=args.fps)
    features.to_csv(out_dir / "features.csv", index=False)

    preds = predict(features, args.model)
    preds = majority_vote_smooth(preds, window=args.smooth_window)
    preds = enforce_min_bout(preds, min_frames=args.min_bout_frames)
    preds.to_csv(out_dir / "predictions.csv", index=False)
    print(f"predictions -> {out_dir / 'predictions.csv'}")

    if args.video:
        from pipeline.visualize import annotate_video, annotate_frames

        annotate_video(
            args.video, resolved, str(out_dir / "annotated.mp4"), predictions=preds, review_frames=review
        )
        print(f"annotated video -> {out_dir / 'annotated.mp4'}")

        if args.export_frames:
            frames_dir = out_dir / "frames"
            written = annotate_frames(
                args.video,
                resolved,
                str(frames_dir),
                predictions=preds,
                review_frames=review,
                image_ext=args.frame_ext,
                every_n=args.frame_every_n,
            )
            print(f"{len(written)} annotated frame image(s) -> {frames_dir}")
    elif args.export_frames:
        print("--export-frames was given but --video wasn't, nothing to draw the overlay onto, skipping.")

    if args.ground_truth:
        gt = pd.read_csv(args.ground_truth)
        report_path = out_dir / "eval_report.txt"
        with open(report_path, "w") as fh:
            fh.write("frame-level:\n")
            fh.write(frame_level_report(preds, gt)["report"])
            fh.write("\nevent-level (bouts):\n")
            fh.write(event_level_report(preds, gt).to_string(index=False))
        print(f"evaluation -> {report_path}")


if __name__ == "__main__":
    main()
