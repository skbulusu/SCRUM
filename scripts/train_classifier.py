from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.io_utils import load_tracking
from pipeline.identity_resolution import ResolutionConfig, resolve_identities
from pipeline.features import compute_features
from pipeline.behavior_classifier import train


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracking", required=True, help="SLEAP analysis .h5 or DeepLabCut multi-animal .h5")
    ap.add_argument("--labels", required=True, help="CSV with columns: frame,behavior")
    ap.add_argument("--model-out", required=True, help="where to save the trained model (.joblib)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument(
        "--contact-radius-px", type=float, default=40.0, help="see identity_resolution.ResolutionConfig"
    )
    ap.add_argument(
        "--review-out", default=None, help="optional path to write flagged occlusion segments as CSV"
    )
    args = ap.parse_args()

    print(f"Loading tracking: {args.tracking}")
    tidy = load_tracking(args.tracking)

    print("Resolving identities through occlusion...")
    resolved, review = resolve_identities(tidy, ResolutionConfig(contact_radius_px=args.contact_radius_px))
    if args.review_out:
        review.to_csv(args.review_out, index=False)
        print(
            f"  wrote {len(review)} flagged segment(s) to {args.review_out}, proofread these before trusting labels near them"
        )

    print("Computing features...")
    features = compute_features(resolved, fps=args.fps)

    print(f"Loading labels: {args.labels}")
    labels = pd.read_csv(args.labels)

    print("Training classifier...")
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    result = train(features, labels, args.model_out)
    print(f"Saved model to {args.model_out} ({result['n_train']} train / {result['n_test']} test frames)\n")
    print(result["report"])


if __name__ == "__main__":
    main()
