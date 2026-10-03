from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

DEFAULT_BEHAVIOR_CLASSES = ["none", "sniffing", "biting", "mounting"]


def _make_model(model: str, random_state: int):
    if model == "rf":

        return RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=random_state,
        )
    if model == "hgb":

        return HistGradientBoostingClassifier(
            max_iter=300, class_weight="balanced", random_state=random_state
        )
    raise ValueError(f"model must be 'rf' or 'hgb', got {model!r}")


def train(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    model_path: str,
    behavior_classes: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 0,
    model: str = "hgb",
):
    behavior_classes = behavior_classes or DEFAULT_BEHAVIOR_CLASSES
    merged = features.merge(labels, on="frame", how="inner")
    if merged.empty:
        raise ValueError(
            "No overlapping frames between features and labels, check the 'frame' columns line up."
        )

    feature_cols = [c for c in features.columns if c != "frame"]
    X = merged[feature_cols].fillna(0.0)
    y = merged["behavior"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None
    )

    clf = _make_model(model, random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    joblib.dump(
        {"model": clf, "feature_cols": feature_cols, "behavior_classes": behavior_classes}, model_path
    )
    return {
        "report": report,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "feature_cols": feature_cols,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def _runs_of(seq: list) -> list[list]:
    runs = []
    start = 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[start]:
            runs.append([start, i - 1, seq[start]])
            start = i
    return runs


def smooth_predictions(
    predictions: pd.DataFrame, min_run_frames: int = 5, max_gap_frames: int | None = None
) -> pd.DataFrame:
    max_gap_frames = min_run_frames if max_gap_frames is None else max_gap_frames
    behavior = predictions["behavior"].tolist()
    if len(behavior) == 0:
        return predictions.copy()

    runs = _runs_of(behavior)

    i = 1
    while i < len(runs) - 1:
        s, e, lab = runs[i]
        prev_lab, next_lab = runs[i - 1][2], runs[i + 1][2]
        if lab == "none" and prev_lab == next_lab and prev_lab != "none" and (e - s + 1) <= max_gap_frames:
            runs[i - 1 : i + 2] = [[runs[i - 1][0], runs[i + 1][1], prev_lab]]

        else:
            i += 1

    changed = True
    while changed and len(runs) > 1:
        changed = False
        for idx, (s, e, _lab) in enumerate(runs):
            if e - s + 1 >= min_run_frames:
                continue
            prev_len = (runs[idx - 1][1] - runs[idx - 1][0] + 1) if idx > 0 else -1
            next_len = (runs[idx + 1][1] - runs[idx + 1][0] + 1) if idx < len(runs) - 1 else -1
            if prev_len < 0 and next_len < 0:
                break
            if prev_len >= next_len:
                runs[idx - 1][1] = e
            else:
                runs[idx + 1][0] = s
            del runs[idx]
            changed = True
            break

    smoothed = behavior[:]
    for s, e, lab in runs:
        smoothed[s : e + 1] = [lab] * (e - s + 1)

    out = predictions.copy()
    out["behavior"] = smoothed
    return out


def predict(features: pd.DataFrame, model_path: str, smooth_min_run_frames: int = 5) -> pd.DataFrame:
    bundle = joblib.load(model_path)
    clf, feature_cols = bundle["model"], bundle["feature_cols"]

    X = features[feature_cols].fillna(0.0)
    proba = clf.predict_proba(X)
    pred = clf.classes_[np.argmax(proba, axis=1)]
    confidence = proba.max(axis=1)

    out = pd.DataFrame({"frame": features["frame"].to_numpy(), "behavior": pred, "confidence": confidence})
    if smooth_min_run_frames and smooth_min_run_frames > 1:
        out = smooth_predictions(out, min_run_frames=smooth_min_run_frames)
    return out
