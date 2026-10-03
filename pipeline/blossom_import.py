from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

BODYPART_RENAME = {"top": "nose", "tail": "tailbase"}

EXPERIMENT_CLASSES = {
    "behavior": ["none", "aggression", "social", "nonsocial"],
    "aggression": [
        "none",
        "lateralthreat",
        "keepdown",
        "clinch",
        "uprightposture",
        "freezing",
        "bite",
        "chase",
    ],
    "sniffing_biting_mounting": ["none", "sniffing", "biting", "mounting"],
}

SBM_CAVEATS = {
    "biting": "direct match: S2 'bite'",
    "sniffing": "approximate: S1 'social' (any non-aggressive contact, not sniffing specifically)",
    "mounting": "weak proxy: S2 'clinch'/'keepdown' (aggressive grappling/pinning, not verified mounting, this paradigm has no real mounting labels)",
}


def dlc_csv_to_h5(csv_path: str | Path, h5_path: str | Path) -> int:
    df = pd.read_csv(csv_path, header=[0, 1, 2, 3], index_col=0)
    df.columns = df.columns.set_levels([BODYPART_RENAME.get(b, b) for b in df.columns.levels[2]], level=2)
    Path(h5_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(h5_path, key="df", mode="w")
    return len(df)


def parse_caltech_annot(path: str | Path) -> dict[str, list[tuple[int, int, str]]]:
    text = Path(path).read_text(encoding="utf-8")
    out: dict[str, list[tuple[int, int, str]]] = {"S1": [], "S2": []}
    for section in ("S1", "S2"):
        other = "S2" if section == "S1" else None
        stop = r"S2:|$" if section == "S1" else r"$"
        m = re.search(rf"{section}:\s+start\s+end\s+type\s*\n-+\n(.*?)(?={stop})", text, re.DOTALL)
        if not m:
            continue
        for line in m.group(1).strip().splitlines():
            seg = re.match(r"\s*(\d+)\s+(\d+)\s+(\w+)", line.strip())
            if seg:
                out[section].append((int(seg.group(1)), int(seg.group(2)), seg.group(3).lower()))
    return out


def annot_to_labels(annot_path: str | Path, n_frames: int, experiment: str = "behavior") -> pd.DataFrame:
    if experiment not in EXPERIMENT_CLASSES:
        raise ValueError(f"experiment must be one of {list(EXPERIMENT_CLASSES)}, got {experiment!r}")
    section = "S1" if experiment == "behavior" else "S2"
    segments = parse_caltech_annot(annot_path)[section]

    behavior = np.full(n_frames + 1, "none", dtype=object)
    for start, end, label in segments:
        if label == "base":
            continue
        start_idx = max(1, start)
        end_idx = min(n_frames, end)
        behavior[start_idx : end_idx + 1] = label

    frames = np.arange(1, n_frames + 1)
    return pd.DataFrame({"frame": frames, "behavior": behavior[1:]})


def annot_to_sbm_labels(annot_path: str | Path, n_frames: int) -> pd.DataFrame:
    segments = parse_caltech_annot(annot_path)
    behavior = np.full(n_frames + 1, "none", dtype=object)

    for start, end, label in segments["S1"]:
        if label == "social":
            s, e = max(1, start), min(n_frames, end)
            behavior[s : e + 1] = "sniffing"

    for start, end, label in segments["S2"]:
        s, e = max(1, start), min(n_frames, end)
        if label == "bite":
            behavior[s : e + 1] = "biting"
        elif label in ("clinch", "keepdown"):
            behavior[s : e + 1] = "mounting"

    frames = np.arange(1, n_frames + 1)
    return pd.DataFrame({"frame": frames, "behavior": behavior[1:]})


def convert_video(
    data_dir: str | Path, video_id: int, out_dir: str | Path, experiment: str = "behavior"
) -> tuple[Path, Path]:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_matches = sorted((data_dir / "data" / "dlc_csv").glob(f"{video_id}DLC*.csv"))
    if not csv_matches:
        raise FileNotFoundError(f"No DLC csv for video {video_id} under {data_dir/'data'/'dlc_csv'}")
    annot_path = data_dir / "data" / "annotations" / f"{video_id}_annot.txt"
    if not annot_path.exists():
        raise FileNotFoundError(f"No annotation file for video {video_id}: {annot_path}")

    h5_path = out_dir / f"{video_id}.h5"
    n_frames = dlc_csv_to_h5(csv_matches[0], h5_path)

    if experiment == "sniffing_biting_mounting":
        labels = annot_to_sbm_labels(annot_path, n_frames)
    else:
        labels = annot_to_labels(annot_path, n_frames, experiment=experiment)
    labels_path = out_dir / f"{video_id}_labels.csv"
    labels.to_csv(labels_path, index=False)

    return h5_path, labels_path


def available_video_ids(data_dir: str | Path) -> list[int]:
    data_dir = Path(data_dir)
    csv_ids = {int(p.stem.split("DLC")[0]) for p in (data_dir / "data" / "dlc_csv").glob("*DLC*.csv")}
    annot_ids = {int(p.stem.split("_")[0]) for p in (data_dir / "data" / "annotations").glob("*_annot.txt")}
    return sorted(csv_ids & annot_ids)
