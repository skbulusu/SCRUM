from __future__ import annotations

import pandas as pd


def load_sleap_h5(path: str) -> pd.DataFrame:
    import h5py
    import numpy as np

    with h5py.File(path, "r") as f:
        track_names = [t.decode() if isinstance(t, bytes) else str(t) for t in f["track_names"][:]]
        node_names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["node_names"][:]]
        tracks = f["tracks"][:]
        scores = f["point_scores"][:] if "point_scores" in f else np.ones(tracks.shape[:-1])

    n_frames, n_tracks, n_nodes, _ = tracks.shape
    rows = []
    for frame in range(n_frames):
        for t in range(n_tracks):
            for n in range(n_nodes):
                x, y = tracks[frame, t, n]
                if np.isnan(x) or np.isnan(y):
                    continue
                rows.append(
                    {
                        "frame": frame,
                        "track_id": track_names[t],
                        "keypoint": node_names[n],
                        "x": float(x),
                        "y": float(y),
                        "score": float(scores[frame, t, n]),
                    }
                )
    return pd.DataFrame(rows)


def load_dlc_h5(path: str) -> pd.DataFrame:
    df = pd.read_hdf(path)
    scorer = df.columns.get_level_values(0)[0]
    df = df[scorer]

    long = df.stack(level=[0, 1], future_stack=True)
    long.index.names = ["frame", "track_id", "keypoint"]
    long = long.reset_index()
    if "likelihood" not in long.columns:
        long["likelihood"] = 1.0
    long = long.dropna(subset=["x", "y"]).rename(columns={"likelihood": "score"})
    long["score"] = long["score"].fillna(1.0)
    long["frame"] = long["frame"].astype(int)
    long["track_id"] = long["track_id"].astype(str)
    long["keypoint"] = long["keypoint"].astype(str)
    return long[["frame", "track_id", "keypoint", "x", "y", "score"]].reset_index(drop=True)


def load_tracking(path: str) -> pd.DataFrame:
    if path.endswith(".h5") or path.endswith(".hdf5"):
        import h5py

        with h5py.File(path, "r") as f:
            if "track_names" in f and "tracks" in f:
                return load_sleap_h5(path)
        return load_dlc_h5(path)
    raise ValueError(f"Don't know how to load tracking file: {path}")


def centroids(tidy: pd.DataFrame, body_keypoints: list[str] | None = None) -> pd.DataFrame:
    df = tidy
    if body_keypoints:
        df = df[df["keypoint"].isin(body_keypoints)]
    return (
        df.groupby(["frame", "track_id"])[["x", "y"]]
        .mean()
        .reset_index()
        .rename(columns={"x": "cx", "y": "cy"})
    )
