from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)

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


_HGB_PARAM_DIST = {
    "max_iter": [150, 300, 500, 800],
    "max_depth": [None, 4, 6, 8, 12],
    "learning_rate": [0.02, 0.05, 0.1, 0.2],
    "l2_regularization": [0.0, 0.1, 1.0, 5.0],
    "max_leaf_nodes": [15, 31, 63, 127],
    "min_samples_leaf": [5, 20, 50],
}

_RF_PARAM_DIST = {
    "n_estimators": [200, 400, 800],
    "max_depth": [None, 10, 20, 40],
    "min_samples_leaf": [1, 2, 5, 10],
    "max_features": ["sqrt", "log2", 0.5],
}


def _tune_model(
    model: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series | None,
    n_iter: int,
    cv_folds: int,
    random_state: int,
):
    if model == "hgb":
        base = HistGradientBoostingClassifier(class_weight="balanced", random_state=random_state)
        param_dist = _HGB_PARAM_DIST
    elif model == "rf":
        base = RandomForestClassifier(class_weight="balanced", random_state=random_state)
        param_dist = _RF_PARAM_DIST
    else:
        raise ValueError(f"model must be 'rf' or 'hgb', got {model!r}")

    if groups_train is not None and groups_train.nunique() >= cv_folds:
        cv = list(GroupKFold(n_splits=cv_folds).split(X_train, y_train, groups_train))
    else:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        base,
        param_dist,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def train(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    model_path: str,
    behavior_classes: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 0,
    model: str = "hgb",
    tune: bool = False,
    n_iter: int = 20,
    cv_folds: int = 5,
    oversample_classes: list[str] | None = None,
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


    groups = merged["group"] if "group" in merged.columns else None

    if groups is not None and groups.nunique() > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None
        )
        groups_train = None


    transition_cols = ["frame", "behavior"] + (["group"] if "group" in merged.columns else [])
    transition_source = merged.loc[X_train.index, transition_cols]

    if oversample_classes:
        X_train, y_train = _oversample(X_train, y_train, oversample_classes, random_state)

    best_params = None
    cv_macro_f1 = None
    if tune:
        clf, best_params, cv_macro_f1 = _tune_model(
            model, X_train, y_train, groups_train, n_iter, cv_folds, random_state
        )
    else:
        clf = _make_model(model, random_state)
        clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)


    classes_ = list(clf.classes_)
    transition_matrix = fit_transition_matrix(transition_source, classes_)

    joblib.dump(
        {
            "model": clf,
            "feature_cols": feature_cols,
            "behavior_classes": behavior_classes,
            "transition_matrix": transition_matrix,
            "transition_classes": classes_,
        },
        model_path,
    )
    result = {
        "report": report,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "feature_cols": feature_cols,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "grouped_split": groups is not None and groups.nunique() > 1,
    }
    if best_params is not None:
        result["best_params"] = best_params
        result["cv_macro_f1"] = cv_macro_f1
    return result


def _oversample(
    X: pd.DataFrame, y: pd.Series, classes: list[str], random_state: int
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(random_state)
    counts = y.value_counts()
    target_classes = [c for c in classes if c in counts.index]
    if not target_classes:
        return X, y
    target_n = counts[target_classes].max()

    extra_X, extra_y = [], []
    for c in target_classes:
        idx = y[y == c].index
        n_needed = target_n - len(idx)
        if n_needed <= 0:
            continue
        sampled = rng.choice(idx, size=n_needed, replace=True)
        extra_X.append(X.loc[sampled])
        extra_y.append(y.loc[sampled])

    if not extra_X:
        return X, y
    X_out = pd.concat([X] + extra_X, ignore_index=True)
    y_out = pd.concat([y] + extra_y, ignore_index=True)
    return X_out, y_out


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


def fit_transition_matrix(
    labels: pd.DataFrame, classes: list[str], smoothing: float = 1.0
) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    counts = np.full((n, n), smoothing)
    group_col = labels["group"] if "group" in labels.columns else pd.Series(0, index=labels.index)
    for _, g in labels.assign(_group=group_col).sort_values("frame").groupby("_group"):
        seq = g["behavior"].tolist()
        for i in range(len(seq) - 1):
            cur, nxt = seq[i], seq[i + 1]
            if cur in idx and nxt in idx:
                counts[idx[cur], idx[nxt]] += 1
    return counts / counts.sum(axis=1, keepdims=True)


def viterbi_smooth(proba: np.ndarray, classes: list[str], transition: np.ndarray) -> np.ndarray:
    n_frames, n_classes = proba.shape
    if n_frames == 0:
        return np.array([], dtype=object)
    log_proba = np.log(np.clip(proba, 1e-9, 1.0))
    log_trans = np.log(np.clip(transition, 1e-9, 1.0))

    log_delta = np.zeros((n_frames, n_classes))
    backptr = np.zeros((n_frames, n_classes), dtype=int)
    log_delta[0] = log_proba[0]

    for t in range(1, n_frames):
        scores = log_delta[t - 1][:, None] + log_trans
        backptr[t] = np.argmax(scores, axis=0)
        log_delta[t] = scores[backptr[t], np.arange(n_classes)] + log_proba[t]

    path = np.zeros(n_frames, dtype=int)
    path[-1] = int(np.argmax(log_delta[-1]))
    for t in range(n_frames - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]

    classes_arr = np.array(classes, dtype=object)
    return classes_arr[path]


def load_model_bundle(model_path) -> dict:
    try:
        return joblib.load(model_path)
    except (ModuleNotFoundError, AttributeError, ImportError) as e:
        import sklearn

        raise RuntimeError(
            f"Could not load the model {Path(model_path).name} with scikit-learn {sklearn.__version__} ({e}). "
            "These models were saved with scikit-learn 1.7. Install it with: pip install \"scikit-learn>=1.7,<1.8\""
        ) from e

def predict(
    features: pd.DataFrame,
    model_path: str,
    smooth_min_run_frames: int = 5,
    smoothing: str = "runlength",
) -> pd.DataFrame:
    bundle = load_model_bundle(model_path)
    clf, feature_cols = bundle["model"], bundle["feature_cols"]

    X = features[feature_cols].fillna(0.0)
    proba = clf.predict_proba(X)
    pred = clf.classes_[np.argmax(proba, axis=1)]
    confidence = proba.max(axis=1)

    out = pd.DataFrame({"frame": features["frame"].to_numpy(), "behavior": pred, "confidence": confidence})

    if smoothing == "viterbi":
        transition = bundle.get("transition_matrix")
        transition_classes = bundle.get("transition_classes")
        if transition is None or transition_classes != list(clf.classes_):
            print(
                "warning: this model has no usable saved transition matrix "
                "(trained before viterbi smoothing was added), falling back to runlength smoothing."
            )
            smoothing = "runlength"
        else:
            out["behavior"] = viterbi_smooth(proba, transition_classes, transition)
            return out

    if smoothing == "runlength" and smooth_min_run_frames and smooth_min_run_frames > 1:
        out = smooth_predictions(out, min_run_frames=smooth_min_run_frames)
    return out
