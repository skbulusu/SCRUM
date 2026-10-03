from __future__ import annotations

import json

import numpy as np
import pandas as pd

KEYPOINT_NAMES = ["nose", "left_ear", "right_ear", "neck", "left_hip", "right_hip", "tailbase"]
TRACK_IDS = ["resident", "intruder"]
DEFAULT_VOCAB = {"attack": 0, "investigation": 1, "mount": 2, "other": 3}

TO_SBM_LABEL = {"attack": "biting", "investigation": "sniffing", "mount": "mounting", "other": "none"}


def load_calms21_split(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def iter_sequences(data: dict):
    for _top_key, sequences in data.items():
        if not isinstance(sequences, dict):
            continue
        for seq_name, seq in sequences.items():
            yield seq_name, seq


def sequence_to_tidy(seq: dict) -> pd.DataFrame:
    keypoints = np.asarray(seq["keypoints"])
    scores = np.asarray(seq["scores"]) if seq.get("scores") is not None else None
    if keypoints.ndim != 4 or keypoints.shape[1:] != (len(TRACK_IDS), 2, len(KEYPOINT_NAMES)):
        raise ValueError(
            f"unexpected keypoints shape {keypoints.shape} - this module expects "
            f"(n_frames, {len(TRACK_IDS)}, 2, {len(KEYPOINT_NAMES)}) per the CalMS21 readme. "
            "Run scripts/inspect_calms21.py on your real file and compare before assuming "
            "this module's documented shape is still right."
        )
    n_frames = keypoints.shape[0]
    n_mice, n_kp = len(TRACK_IDS), len(KEYPOINT_NAMES)

    frame = np.repeat(np.arange(n_frames), n_mice * n_kp)
    track_id = np.tile(np.repeat(TRACK_IDS, n_kp), n_frames)
    keypoint = np.tile(KEYPOINT_NAMES, n_frames * n_mice)
    x = keypoints[:, :, 0, :].reshape(-1)
    y = keypoints[:, :, 1, :].reshape(-1)
    score = scores.reshape(-1) if scores is not None else np.ones(n_frames * n_mice * n_kp)

    return pd.DataFrame(
        {"frame": frame, "track_id": track_id, "keypoint": keypoint, "x": x, "y": y, "score": score}
    )


def sequence_to_labels(seq: dict, vocab: dict | None = None, as_sbm: bool = False) -> pd.DataFrame:
    annotations = np.asarray(seq["annotations"])
    seq_vocab = (seq.get("metadata") or {}).get("vocab") or vocab or DEFAULT_VOCAB
    idx_to_name = {v: k for k, v in seq_vocab.items()}
    names = [idx_to_name.get(int(a), f"unknown_{int(a)}") for a in annotations]
    if as_sbm:
        names = [TO_SBM_LABEL.get(n, n) for n in names]
    return pd.DataFrame({"frame": np.arange(len(annotations)), "behavior": names})
