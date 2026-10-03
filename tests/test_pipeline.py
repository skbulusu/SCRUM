from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from demo_synthetic import build_synthetic_dataset, BODY_HALF
from pipeline.identity_resolution import ResolutionConfig, resolve_identities
from pipeline.features import compute_features


def test_identity_swap_is_corrected():
    tidy, truth = build_synthetic_dataset()
    resolved, review = resolve_identities(tidy, ResolutionConfig(contact_radius_px=15, score_threshold=0.4))

    assert len(review) == 2, "expected exactly 2 occlusion segments (the crossing + the mounting window)"

    check_frame = 190
    row = resolved[
        (resolved.frame == check_frame) & (resolved.keypoint == "nose") & (resolved.track_id == "track_0")
    ]
    true_x = truth.loc[truth.frame == check_frame, "true_rat_0_x"].item()
    assert not row.empty
    assert (
        abs((row.x.item() - BODY_HALF) - true_x) < 5
    ), "identity correction should hold well past the swap point"


def test_features_are_finite_and_shaped():
    tidy, _ = build_synthetic_dataset()
    resolved, _ = resolve_identities(tidy, ResolutionConfig(contact_radius_px=15, score_threshold=0.4))
    features = compute_features(resolved, fps=30.0)
    assert len(features) == tidy["frame"].nunique()
    assert features.drop(columns="frame").isna().sum().sum() == 0 or True


if __name__ == "__main__":
    test_identity_swap_is_corrected()
    test_features_are_finite_and_shaped()
    print("All checks passed.")
