from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TIDY_COLUMNS = ["frame", "track_id", "keypoint", "x", "y", "score"]


KEYPOINT_ALIASES = {
    "nose": "nose",
    "snout": "nose",
    "nose_tip": "nose",
    "nosetip": "nose",
    "head": "nose",
    "left_ear": "left_ear",
    "leftear": "left_ear",
    "ear_left": "left_ear",
    "earl": "left_ear",
    "right_ear": "right_ear",
    "rightear": "right_ear",
    "ear_right": "right_ear",
    "earr": "right_ear",
    "neck": "neck",
    "left_hip": "left_hip",
    "hip_left": "left_hip",
    "lefthip": "left_hip",
    "right_hip": "right_hip",
    "hip_right": "right_hip",
    "righthip": "right_hip",
    "tailbase": "tailbase",
    "tail_base": "tailbase",
    "tail_root": "tailbase",
    "tailroot": "tailbase",
    "base_of_tail": "tailbase",
    "tail_start": "tailbase",
    "tailstart": "tailbase",
}

REQUIRED_KEYPOINTS = ("nose", "tailbase")


def map_keypoint_names(tidy: pd.DataFrame, mapping: dict[str, str] | None = None) -> pd.DataFrame:
    mapping = {str(k).lower(): v for k, v in (mapping or {}).items()}

    def convert(name: str) -> str:
        key = str(name).lower()
        if key in mapping:
            return mapping[key]
        return KEYPOINT_ALIASES.get(key, name)

    out = tidy.copy()
    out["keypoint"] = out["keypoint"].map(convert)
    missing = [k for k in REQUIRED_KEYPOINTS if k not in set(out["keypoint"])]
    if missing:
        raise ValueError(
            f"Could not find keypoint(s) {missing} after name mapping. The file contains: "
            f"{sorted(set(tidy['keypoint']))}. Pass a mapping such as "
            f"{{'<your nose node>': 'nose', '<your tail base node>': 'tailbase'}}."
        )
    return out


def _finalize(tidy: pd.DataFrame, max_tracks: int | None = 2) -> pd.DataFrame:
    tidy = tidy.dropna(subset=["x", "y"]).copy()
    tidy["frame"] = tidy["frame"].astype(int)
    if "score" not in tidy.columns:
        tidy["score"] = np.nan
    if max_tracks is not None and tidy["track_id"].nunique() > max_tracks:
        keep = tidy.groupby("track_id")["frame"].nunique().nlargest(max_tracks).index
        tidy = tidy[tidy["track_id"].isin(keep)]
    return tidy[TIDY_COLUMNS].sort_values(["frame", "track_id", "keypoint"]).reset_index(drop=True)


def load_dlc_csv(path: str | Path, track_id: str | None = None) -> pd.DataFrame:
    path = Path(path)
    with open(path, newline="") as fh:
        head = [next(fh, "") for _ in range(5)]
    second_cell = head[1].split(",")[0].strip().lower()
    multi = second_cell == "individuals"
    header_rows = [0, 1, 2, 3] if multi else [0, 1, 2]
    raw = pd.read_csv(path, header=header_rows, index_col=0)

    frames = np.arange(len(raw)) if not np.issubdtype(raw.index.dtype, np.integer) else raw.index.to_numpy()
    rows = []
    for col in raw.columns:
        if multi:
            _scorer, individual, bodypart, coord = col
        else:
            _scorer, bodypart, coord = col
            individual = track_id or path.stem
        if coord not in ("x", "y", "likelihood"):
            continue
        rows.append((individual, bodypart, coord, raw[col].to_numpy()))

    parts = {}
    for individual, bodypart, coord, values in rows:
        parts.setdefault((individual, bodypart), {})[coord] = values
    frames_out = []
    for (individual, bodypart), d in parts.items():
        if "x" not in d or "y" not in d:
            continue
        frames_out.append(
            pd.DataFrame(
                {
                    "frame": frames,
                    "track_id": str(individual),
                    "keypoint": str(bodypart),
                    "x": d["x"],
                    "y": d["y"],
                    "score": d.get("likelihood", np.full(len(frames), np.nan)),
                }
            )
        )
    if not frames_out:
        raise ValueError(f"No x/y columns found in DeepLabCut file {path}")
    return pd.concat(frames_out, ignore_index=True)


def load_sleap_h5(path: str | Path) -> pd.DataFrame:
    import h5py

    with h5py.File(path, "r") as f:
        if "tracks" not in f:
            raise ValueError(
                f"{path} has no 'tracks' dataset. Export from SLEAP with File > Export Analysis HDF5."
            )
        tracks = np.asarray(f["tracks"])
        node_names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["node_names"][:]]
        if "track_names" in f:
            track_names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["track_names"][:]]
        else:
            track_names = [f"track_{i}" for i in range(tracks.shape[0])]
        point_scores = np.asarray(f["point_scores"]) if "point_scores" in f else None

    if tracks.ndim != 4 or tracks.shape[1] != 2:
        raise ValueError(f"Unexpected SLEAP tracks shape {tracks.shape}, expected (tracks, 2, nodes, frames).")
    n_tracks, _, n_nodes, n_frames = tracks.shape
    if len(node_names) != n_nodes:
        node_names = [f"node_{i}" for i in range(n_nodes)]
    if len(track_names) != n_tracks:
        track_names = [f"track_{i}" for i in range(n_tracks)]

    frames = np.arange(n_frames)
    out = []
    for ti in range(n_tracks):
        for ni in range(n_nodes):
            out.append(
                pd.DataFrame(
                    {
                        "frame": frames,
                        "track_id": track_names[ti],
                        "keypoint": node_names[ni],
                        "x": tracks[ti, 0, ni, :],
                        "y": tracks[ti, 1, ni, :],
                        "score": point_scores[ti, ni, :] if point_scores is not None else np.nan,
                    }
                )
            )
    return pd.concat(out, ignore_index=True)


def load_sleap_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    frame_col = next((c for c in ("frame_idx", "frame") if c in df.columns), None)
    if frame_col is None or "track" not in df.columns:
        raise ValueError(f"{path} does not look like a SLEAP analysis CSV (needs 'track' and 'frame_idx' columns).")
    nodes = sorted({c[:-2] for c in df.columns if c.endswith(".x") and f"{c[:-2]}.y" in df.columns})
    if not nodes:
        raise ValueError(f"No '<node>.x' / '<node>.y' columns found in {path}")
    out = []
    for node in nodes:
        score_col = f"{node}.score"
        out.append(
            pd.DataFrame(
                {
                    "frame": df[frame_col].to_numpy(),
                    "track_id": df["track"].astype(str).to_numpy(),
                    "keypoint": node,
                    "x": df[f"{node}.x"].to_numpy(),
                    "y": df[f"{node}.y"].to_numpy(),
                    "score": df[score_col].to_numpy() if score_col in df.columns else np.nan,
                }
            )
        )
    return pd.concat(out, ignore_index=True)


def load_slp(path: str | Path) -> pd.DataFrame:
    try:
        import sleap_io as sio
    except ImportError as e:
        raise ImportError(
            "Reading .slp files needs the sleap-io package (pip install sleap-io). "
            "Alternatively export from SLEAP with File > Export Analysis HDF5 and open that file."
        ) from e
    labels = sio.load_slp(str(path))
    node_names = [n.name for n in labels.skeleton.nodes]
    rows = []
    for lf in labels.labeled_frames:
        for inst in lf.instances:
            track = inst.track.name if inst.track is not None else "track_0"
            pts = np.asarray(inst.numpy())
            for ni, name in enumerate(node_names):
                rows.append((lf.frame_idx, track, name, pts[ni, 0], pts[ni, 1]))
    return pd.DataFrame(rows, columns=["frame", "track_id", "keypoint", "x", "y"]).assign(score=np.nan)


def detect_format(path: str | Path) -> str:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".h5", ".hdf5"):
        return "sleap_h5"
    if suffix == ".slp":
        return "slp"
    if suffix == ".csv":
        with open(p, newline="") as fh:
            first = fh.readline().strip().lower()
            second = fh.readline().strip().lower()
        if first.startswith("scorer") or second.startswith(("bodyparts", "individuals")):
            return "dlc"
        header = first.split(",")
        if "track" in header and ("frame_idx" in header or "frame" in header):
            return "sleap_csv"
        if "keypoint" in header and "track_id" in header:
            return "tidy"
        return "tidy"
    raise ValueError(f"Don't know how to import {p.name}")


def load_pose_file(path: str | Path, mapping: dict[str, str] | None = None, track_id: str | None = None) -> pd.DataFrame:
    fmt = detect_format(path)
    if fmt == "dlc":
        tidy = load_dlc_csv(path, track_id=track_id)
    elif fmt == "sleap_h5":
        tidy = load_sleap_h5(path)
    elif fmt == "sleap_csv":
        tidy = load_sleap_csv(path)
    elif fmt == "slp":
        tidy = load_slp(path)
    else:
        tidy = pd.read_csv(path)
        missing = {"frame", "track_id", "keypoint", "x", "y"} - set(tidy.columns)
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}")
    return _finalize(map_keypoint_names(tidy, mapping), max_tracks=2)


def load_pose_files(paths: list[str | Path], mapping: dict[str, str] | None = None) -> pd.DataFrame:
    if len(paths) == 1:
        return load_pose_file(paths[0], mapping)
    frames = []
    for i, p in enumerate(paths):
        t = load_pose_file(p, mapping, track_id=f"track_{i}")
        t["track_id"] = f"track_{i}"
        frames.append(t)
    return _finalize(pd.concat(frames, ignore_index=True), max_tracks=None)
