from __future__ import annotations

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .smoothing import bouts_from_frames


def frame_level_report(predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> dict:
    merged = predictions.merge(ground_truth, on="frame", suffixes=("_pred", "_true"), how="inner")
    if merged.empty:
        raise ValueError("No overlapping frames between predictions and ground truth.")
    report = classification_report(merged["behavior_true"], merged["behavior_pred"], zero_division=0)
    labels = sorted(set(merged["behavior_true"]) | set(merged["behavior_pred"]))
    cm = confusion_matrix(merged["behavior_true"], merged["behavior_pred"], labels=labels)
    return {"report": report, "confusion_matrix": cm, "labels": labels, "n_frames": len(merged)}


def event_level_report(
    predictions: pd.DataFrame, ground_truth: pd.DataFrame, iou_threshold: float = 0.3
) -> pd.DataFrame:
    pred_bouts = bouts_from_frames(predictions)
    true_bouts = bouts_from_frames(ground_truth)

    rows = []
    for behavior in sorted(set(pred_bouts["behavior"]) | set(true_bouts["behavior"])):
        p = pred_bouts[pred_bouts["behavior"] == behavior]
        t = true_bouts[true_bouts["behavior"] == behavior]
        matched_true = set()
        tp = 0
        for _, pb in p.iterrows():
            for ti, tb in t.iterrows():
                if ti in matched_true:
                    continue
                inter = max(0, min(pb.frame_end, tb.frame_end) - max(pb.frame_start, tb.frame_start) + 1)
                union = (pb.frame_end - pb.frame_start + 1) + (tb.frame_end - tb.frame_start + 1) - inter
                if union > 0 and inter / union >= iou_threshold:
                    tp += 1
                    matched_true.add(ti)
                    break
        fp = len(p) - tp
        fn = len(t) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "behavior": behavior,
                "n_pred_bouts": len(p),
                "n_true_bouts": len(t),
                "tp": tp,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
            }
        )
    return pd.DataFrame(rows)
