from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_VERSION = 2
TRACK_COLUMNS = ["frame", "track_id", "keypoint", "x", "y"]


@dataclass
class DetectionResult:
    video_path: str = ""
    fps: float = 30.0
    n_frames: int = 0
    width: int = 0
    height: int = 0
    source: str = ""
    mode: str = "scrum"
    tracks: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=TRACK_COLUMNS))
    predictions: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["frame", "behavior", "confidence"]))

    outlines: dict = field(default_factory=dict)
    merged: np.ndarray | None = None


    def save(self, path: str | Path) -> None:
        path = str(path)
        t = self.tracks
        p = self.predictions
        frames, tids, flags, ncont, clen, pts = [], [], [], [], [], []
        for f in sorted(self.outlines):
            for entry in self.outlines[f]:
                frames.append(f)
                tids.append(entry["track_id"])
                flags.append(int(bool(entry.get("merged"))) | (int(bool(entry.get("split"))) << 1))
                conts = [np.asarray(c).reshape(-1, 2).astype(np.int32) for c in entry["contours"] if c is not None and len(c)]
                ncont.append(len(conts))
                for c in conts:
                    clen.append(len(c))
                    pts.append(c)
        meta = {
            "version": CACHE_VERSION,
            "video_path": self.video_path,
            "fps": self.fps,
            "n_frames": self.n_frames,
            "width": self.width,
            "height": self.height,
            "source": self.source,
            "mode": self.mode,
            "video_size": os.path.getsize(self.video_path) if self.video_path and os.path.exists(self.video_path) else -1,
        }
        np.savez_compressed(
            path,
            meta=np.array(json.dumps(meta)),
            t_frame=t["frame"].to_numpy(dtype=np.int32) if len(t) else np.zeros(0, np.int32),
            t_id=t["track_id"].astype(str).to_numpy(dtype="U24") if len(t) else np.zeros(0, "U24"),
            t_kp=t["keypoint"].astype(str).to_numpy(dtype="U24") if len(t) else np.zeros(0, "U24"),
            t_x=t["x"].to_numpy(dtype=np.float32) if len(t) else np.zeros(0, np.float32),
            t_y=t["y"].to_numpy(dtype=np.float32) if len(t) else np.zeros(0, np.float32),
            p_frame=p["frame"].to_numpy(dtype=np.int32) if len(p) else np.zeros(0, np.int32),
            p_behavior=p["behavior"].astype(str).to_numpy(dtype="U24") if len(p) else np.zeros(0, "U24"),
            p_conf=(p["confidence"].to_numpy(dtype=np.float32) if "confidence" in p.columns and len(p) else np.full(len(p), np.nan, np.float32)),
            merged=(self.merged if self.merged is not None else np.zeros(0, bool)).astype(bool),
            o_frame=np.array(frames, dtype=np.int32),
            o_tid=np.array(tids, dtype="U24") if tids else np.zeros(0, "U24"),
            o_flags=np.array(flags, dtype=np.uint8),
            o_ncont=np.array(ncont, dtype=np.int32),
            c_len=np.array(clen, dtype=np.int32),
            c_pts=np.concatenate(pts, axis=0) if pts else np.zeros((0, 2), np.int32),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DetectionResult":
        with np.load(str(path), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta.get("version") != CACHE_VERSION:
                raise ValueError("cache was written by a different version of SCRUM")
            tracks = pd.DataFrame(
                {"frame": z["t_frame"], "track_id": z["t_id"], "keypoint": z["t_kp"], "x": z["t_x"].astype(float), "y": z["t_y"].astype(float)}
            )
            preds = pd.DataFrame({"frame": z["p_frame"], "behavior": z["p_behavior"], "confidence": z["p_conf"].astype(float)})
            outlines: dict[int, list[dict]] = {}
            c_pos = 0
            pts_pos = 0
            c_len, c_pts = z["c_len"], z["c_pts"]
            for f, tid, flag, nc in zip(z["o_frame"], z["o_tid"], z["o_flags"], z["o_ncont"]):
                conts = []
                for _ in range(int(nc)):
                    n = int(c_len[c_pos])
                    conts.append(c_pts[pts_pos : pts_pos + n].reshape(-1, 1, 2).astype(np.int32))
                    pts_pos += n
                    c_pos += 1
                outlines.setdefault(int(f), []).append(
                    {"track_id": str(tid), "contours": conts, "merged": bool(flag & 1), "split": bool(flag & 2)}
                )
            merged = z["merged"].astype(bool) if z["merged"].size else None
        res = cls(
            video_path=meta["video_path"], fps=meta["fps"], n_frames=meta["n_frames"], width=meta["width"],
            height=meta["height"], source=meta["source"], mode=meta["mode"], tracks=tracks, predictions=preds,
            outlines=outlines, merged=merged,
        )
        res.video_size = meta.get("video_size", -1)
        return res

    def matches_video(self, video_path: str) -> bool:
        size = getattr(self, "video_size", -1)
        return bool(os.path.exists(video_path) and (size < 0 or os.path.getsize(video_path) == size))


def cache_path_for(video_path: str | Path) -> Path:
    p = Path(video_path)
    return p.with_name(p.stem + ".scrum_cache.npz")


def results_dir_for(video_path: str | Path) -> Path:
    return Path(video_path).parent / "scrum_results"


def find_cache(video_path: str | Path) -> Path | None:
    p = Path(video_path)
    for cand in (cache_path_for(p), results_dir_for(p) / (p.stem + ".scrum_cache.npz")):
        if cand.exists():
            return cand
    return None
