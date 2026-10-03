from __future__ import annotations

import pandas as pd


def majority_vote_smooth(predictions: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    out = predictions.copy().sort_values("frame").reset_index(drop=True)
    half = window // 2
    labels = out["behavior"].tolist()
    smoothed = []
    for i in range(len(labels)):
        lo, hi = max(0, i - half), min(len(labels), i + half + 1)
        window_labels = labels[lo:hi]
        smoothed.append(max(set(window_labels), key=window_labels.count))
    out["behavior"] = smoothed
    return out


def enforce_min_bout(
    predictions: pd.DataFrame, min_frames: int = 3, background_label: str = "none"
) -> pd.DataFrame:
    out = predictions.copy().sort_values("frame").reset_index(drop=True)
    labels = out["behavior"].tolist()

    i = 0
    while i < len(labels):
        j = i
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        if labels[i] != background_label and (j - i) < min_frames:
            for k in range(i, j):
                labels[k] = background_label
        i = j

    out["behavior"] = labels
    return out


def bouts_from_frames(predictions: pd.DataFrame, background_label: str = "none") -> pd.DataFrame:
    df = predictions.sort_values("frame").reset_index(drop=True)
    rows = []
    start_idx = 0
    for i in range(1, len(df) + 1):
        if i == len(df) or df.loc[i, "behavior"] != df.loc[start_idx, "behavior"]:
            label = df.loc[start_idx, "behavior"]
            if label != background_label:
                rows.append(
                    {
                        "behavior": label,
                        "frame_start": int(df.loc[start_idx, "frame"]),
                        "frame_end": int(df.loc[i - 1, "frame"]),
                        "n_frames": i - start_idx,
                    }
                )
            start_idx = i
    return pd.DataFrame(rows, columns=["behavior", "frame_start", "frame_end", "n_frames"])
