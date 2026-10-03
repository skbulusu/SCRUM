from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class TrackerConfig:
    n_background_samples: int = 60
    diff_threshold: int = 25
    min_blob_area: int = 200
    morph_kernel_size: int = 7
    max_jump_px: float = 80.0
    contact_radius_px: float = 70.0
    max_missing_before_reacquire: int = 15
    static_artifact_area: float = 900.0
    static_artifact_frames: int = 4
    static_artifact_jitter_px: float = 14.0
    static_artifact_miss_grace: int = 2
    outline_diff_threshold: int = 12
    outline_morph_kernel_size: int = 3
    outline_roi_pad: int = 25
    contact_gap_px: float = 12.0
    min_reacquire_area: float = 4000.0
    contact_exit_gap_px: float = 60.0
    contact_exit_frames: int = 3
    contact_confirm_gap: int = 8


def build_background(video_path: str, cfg: TrackerConfig) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 1, min(cfg.n_background_samples, total)).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not frames:
        raise IOError(f"Could not read any frames from {video_path}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def _tight_outline_contour(
    diff: np.ndarray, bbox: tuple, anchor_point: tuple[float, float], cfg: TrackerConfig
) -> np.ndarray | None:
    x, y, w, h = bbox
    H, W = diff.shape[:2]
    pad = cfg.outline_roi_pad
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    roi = diff[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    _, tight_mask = cv2.threshold(roi, cfg.outline_diff_threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((cfg.outline_morph_kernel_size, cfg.outline_morph_kernel_size), np.uint8)
    tight_mask = cv2.morphologyEx(tight_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(tight_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= cfg.min_blob_area * 0.3]
    if not contours:
        return None

    ax, ay = anchor_point[0] - x0, anchor_point[1] - y0
    best = max(contours, key=lambda c: cv2.pointPolygonTest(c, (ax, ay), True))
    return best + np.array([[x0, y0]])


def _detect_blobs(gray_frame: np.ndarray, background: np.ndarray, cfg: TrackerConfig) -> list[dict]:
    diff = cv2.absdiff(gray_frame, background)
    _, mask = cv2.threshold(diff, cfg.diff_threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((cfg.morph_kernel_size, cfg.morph_kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < cfg.min_blob_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        centroid = (x + w / 2, y + h / 2)

        m = cv2.moments(c)
        mass_point = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else centroid

        tight_c = _tight_outline_contour(diff, (x, y, w, h), mass_point, cfg)

        blobs.append(
            {
                "bbox": (x, y, w, h),
                "centroid": centroid,
                "display_centroid": mass_point,
                "area": area,
                "contours": [tight_c if tight_c is not None else c],
            }
        )
    return blobs


def _bbox_gap(b1: tuple, b2: tuple) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    dx = max(x2 - (x1 + w1), x1 - (x2 + w2), 0)
    dy = max(y2 - (y1 + h1), y1 - (y2 + h2), 0)
    return float(np.hypot(dx, dy))


def _union_bbox(b1: tuple, b2: tuple) -> tuple:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    x = min(x1, x2)
    y = min(y1, y2)
    w = max(x1 + w1, x2 + w2) - x
    h = max(y1 + h1, y2 + h2) - y
    return (x, y, w, h)


def _resolve_frame(blobs, stable_ids, last_centroids, last_velocity, missing_streak, cfg: TrackerConfig):
    if not last_centroids:

        blobs_sorted = sorted(blobs, key=lambda b: -b["area"])[:2]
        blobs_sorted = sorted(blobs_sorted, key=lambda b: b["centroid"][0])
        out = {}
        for i, tid in enumerate(stable_ids):
            out[tid] = (
                {"status": "tracked", "blob": blobs_sorted[i]}
                if i < len(blobs_sorted)
                else {"status": "missing", "blob": None}
            )
        return out

    trusted_ids = [
        tid
        for tid in stable_ids
        if tid in last_centroids and missing_streak.get(tid, 0) < cfg.max_missing_before_reacquire
    ]
    stale_ids = [tid for tid in stable_ids if tid not in trusted_ids]
    predicted = {tid: last_centroids[tid] for tid in trusted_ids}

    assignment = {}
    remaining_blobs = list(blobs)
    if remaining_blobs and predicted:
        ids_present = list(predicted.keys())
        cost = np.zeros((len(ids_present), len(remaining_blobs)))
        for i, tid in enumerate(ids_present):
            pred = predicted[tid]
            for j, b in enumerate(remaining_blobs):
                cost[i, j] = float(np.hypot(pred[0] - b["centroid"][0], pred[1] - b["centroid"][1]))
        row_idx, col_idx = linear_sum_assignment(cost)
        for r, c in zip(row_idx, col_idx):
            assignment[ids_present[r]] = (remaining_blobs[c], cost[r, c])

    out = {}
    used_blob_ids = set()
    for tid in trusted_ids:
        if tid in assignment:
            blob, dist = assignment[tid]
            if dist <= cfg.max_jump_px:
                out[tid] = {"status": "tracked", "blob": blob}
                used_blob_ids.add(id(blob))

    for tid in trusted_ids:
        if tid in out:
            continue
        pred = predicted.get(tid)
        other_tid = next(
            (t for t in trusted_ids if t != tid and out.get(t, {}).get("blob") is not None), None
        )
        just_seen = missing_streak.get(tid, 0) == 0
        if pred is not None and other_tid is not None and just_seen:
            other_blob = out[other_tid]["blob"]
            merge_dist = float(
                np.hypot(pred[0] - other_blob["centroid"][0], pred[1] - other_blob["centroid"][1])
            )
            if merge_dist <= cfg.contact_radius_px:
                out[tid] = {"status": "contact", "blob": other_blob}
                out[other_tid] = {"status": "contact", "blob": other_blob}
                used_blob_ids.add(id(other_blob))
                continue
        out[tid] = {"status": "missing", "blob": None}

    unresolved = [tid for tid in stable_ids if tid not in out]

    leftover = sorted(
        [b for b in remaining_blobs if id(b) not in used_blob_ids and b["area"] >= cfg.min_reacquire_area],
        key=lambda b: -b["area"],
    )
    for tid, blob in zip(unresolved, leftover):
        out[tid] = {"status": "tracked", "blob": blob}
        used_blob_ids.add(id(blob))

    for tid in stable_ids:
        if tid not in out:
            out[tid] = {"status": "missing", "blob": None}

    return out


def track_video(video_path: str, cfg: TrackerConfig | None = None) -> list[dict]:
    cfg = cfg or TrackerConfig()
    background = build_background(video_path, cfg)

    cap = cv2.VideoCapture(video_path)
    stable_ids = ["rat_0", "rat_1"]
    last_centroids: dict[str, tuple[float, float]] = {}
    last_velocity: dict[str, tuple[float, float]] = {}
    missing_streak: dict[str, int] = {tid: 0 for tid in stable_ids}
    static_anchor: dict[str, tuple[float, float]] = {}
    static_streak: dict[str, int] = {tid: 0 for tid in stable_ids}
    static_miss_gap: dict[str, int] = {tid: 0 for tid in stable_ids}
    in_contact = False
    frozen_contact_blob = None
    contact_exit_streak = 0
    contact_missing_streak = 0

    results = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blobs = _detect_blobs(gray, background, cfg)

        resolved = _resolve_frame(blobs, stable_ids, last_centroids, last_velocity, missing_streak, cfg)

        if resolved["rat_0"]["status"] == "tracked" and resolved["rat_1"]["status"] == "tracked":
            b0, b1 = resolved["rat_0"]["blob"], resolved["rat_1"]["blob"]
            if _bbox_gap(b0["bbox"], b1["bbox"]) <= cfg.contact_gap_px:
                merged_bbox = _union_bbox(b0["bbox"], b1["bbox"])

                merged_blob = {
                    "bbox": merged_bbox,
                    "centroid": (merged_bbox[0] + merged_bbox[2] / 2, merged_bbox[1] + merged_bbox[3] / 2),
                    "area": b0["area"] + b1["area"],
                    "contours": b0["contours"] + b1["contours"],
                }
                resolved["rat_0"] = {"status": "contact", "blob": merged_blob}
                resolved["rat_1"] = {"status": "contact", "blob": merged_blob}

        raw_contact = resolved["rat_0"]["status"] == "contact" and resolved["rat_1"]["status"] == "contact"
        if raw_contact:
            in_contact = True
            frozen_contact_blob = resolved["rat_0"]["blob"]
            contact_exit_streak = 0
            contact_missing_streak = 0
        elif in_contact:
            b0 = resolved["rat_0"]["blob"] if resolved["rat_0"]["status"] == "tracked" else None
            b1 = resolved["rat_1"]["blob"] if resolved["rat_1"]["status"] == "tracked" else None
            contact_missing_streak += 1
            if (
                b0 is not None
                and b1 is not None
                and _bbox_gap(b0["bbox"], b1["bbox"]) > cfg.contact_exit_gap_px
            ):
                contact_exit_streak += 1
            else:
                contact_exit_streak = 0

            if (
                contact_exit_streak >= cfg.contact_exit_frames
                or contact_missing_streak >= cfg.contact_confirm_gap
            ):
                in_contact = False
                frozen_contact_blob = None
                contact_exit_streak = 0
                contact_missing_streak = 0

            else:
                if b0 is not None and b1 is not None:
                    merged_bbox = _union_bbox(b0["bbox"], b1["bbox"])
                    frozen_contact_blob = {
                        "bbox": merged_bbox,
                        "centroid": (
                            merged_bbox[0] + merged_bbox[2] / 2,
                            merged_bbox[1] + merged_bbox[3] / 2,
                        ),
                        "area": b0["area"] + b1["area"],
                        "contours": b0["contours"] + b1["contours"],
                    }

                resolved["rat_0"] = {"status": "contact", "blob": frozen_contact_blob}
                resolved["rat_1"] = {"status": "contact", "blob": frozen_contact_blob}

        frame_tracks = []
        seen_contact_blob_ids = set()
        for tid in stable_ids:
            info = resolved[tid]
            if info["status"] == "tracked":
                blob = info["blob"]

                anchor = static_anchor.get(tid)
                jitter = (
                    float(np.hypot(blob["centroid"][0] - anchor[0], blob["centroid"][1] - anchor[1]))
                    if anchor
                    else None
                )
                if (
                    blob["area"] < cfg.static_artifact_area
                    and jitter is not None
                    and jitter < cfg.static_artifact_jitter_px
                ):
                    static_streak[tid] += 1
                else:
                    static_streak[tid] = 0
                    static_anchor[tid] = blob["centroid"]
                static_miss_gap[tid] = 0

                if static_streak[tid] >= cfg.static_artifact_frames:
                    last_velocity[tid] = (0.0, 0.0)
                    missing_streak[tid] = missing_streak.get(tid, 0) + 1
                    continue

                prev = last_centroids.get(tid)
                was_stale = missing_streak.get(tid, 0) >= cfg.max_missing_before_reacquire
                last_velocity[tid] = (
                    (blob["centroid"][0] - prev[0], blob["centroid"][1] - prev[1])
                    if (prev and not was_stale)
                    else (0.0, 0.0)
                )
                last_centroids[tid] = blob["centroid"]
                missing_streak[tid] = 0
                frame_tracks.append(
                    {
                        "track_id": tid,
                        "bbox": blob["bbox"],
                        "centroid": blob["centroid"],
                        "display_centroid": blob["display_centroid"],
                        "contact": False,
                        "contours": blob["contours"],
                    }
                )
            elif info["status"] == "contact":

                last_velocity[tid] = (0.0, 0.0)
                missing_streak[tid] = 0
                static_streak[tid] = 0
                static_anchor.pop(tid, None)
                static_miss_gap[tid] = 0
                blob_key = id(info["blob"])
                if blob_key not in seen_contact_blob_ids:
                    seen_contact_blob_ids.add(blob_key)
                    frame_tracks.append(
                        {
                            "track_id": "rat_0+rat_1",
                            "bbox": info["blob"]["bbox"],
                            "centroid": info["blob"]["centroid"],
                            "contact": True,
                            "contours": info["blob"].get("contours", []),
                        }
                    )
            else:
                last_velocity[tid] = (0.0, 0.0)
                missing_streak[tid] = missing_streak.get(tid, 0) + 1

                static_miss_gap[tid] += 1
                if static_miss_gap[tid] > cfg.static_artifact_miss_grace:
                    static_streak[tid] = 0
                    static_anchor.pop(tid, None)

        results.append({"frame": frame_idx, "tracks": frame_tracks})
        frame_idx += 1

    cap.release()
    return results


_BRIEF_CONTACT_FRAMES = 10
_HIGH_OVERLAP_RATIO = 0.6
_HIGH_JITTER_PX = 6.0


def _classify_bout(
    n_frames: int, overlap_ratio: float | None, jitter: float, mean_motion: float
) -> tuple[str, str]:
    if (
        overlap_ratio is not None
        and overlap_ratio < _HIGH_OVERLAP_RATIO
        and n_frames >= _BRIEF_CONTACT_FRAMES
    ):
        return (
            "possible mounting",
            f"sustained, high visual overlap (overlap_ratio={overlap_ratio:.2f}, {n_frames} frames)",
        )
    if jitter >= _HIGH_JITTER_PX:
        return (
            "possible vigorous contact (biting/fighting?)",
            f"a lot of relative motion during contact (jitter={jitter:.1f}px, {n_frames} frames)",
        )
    if n_frames <= _BRIEF_CONTACT_FRAMES:
        return ("possible sniffing/investigating", f"brief, low-motion contact ({n_frames} frames)")
    return ("sustained contact (type unclear)", f"{n_frames} frames, no strong signal either way")


def label_contact_bouts(results: list[dict]) -> list[dict]:
    bouts = []
    last_area = {"rat_0": None, "rat_1": None}
    current = None

    def _finalize(b):
        n = len(b["areas"])
        avg_area = sum(b["areas"]) / n
        pre_sum = (b["pre_area_0"] + b["pre_area_1"]) if (b["pre_area_0"] and b["pre_area_1"]) else None
        overlap_ratio = (avg_area / pre_sum) if pre_sum else None
        cents = b["centroids"]
        if len(cents) >= 2:
            diffs = [
                float(np.hypot(cents[i][0] - cents[i - 1][0], cents[i][1] - cents[i - 1][1]))
                for i in range(1, len(cents))
            ]
            jitter = float(np.std(diffs))
            mean_motion = float(np.mean(diffs))
        else:
            jitter = mean_motion = 0.0
        label, note = _classify_bout(n, overlap_ratio, jitter, mean_motion)
        return {
            "frame_start": b["frame_start"],
            "frame_end": b["frame_end"],
            "n_frames": n,
            "overlap_ratio": overlap_ratio,
            "jitter_px": jitter,
            "mean_motion_px": mean_motion,
            "label": label,
            "note": note,
        }

    for r in results:
        contact_track = next((t for t in r["tracks"] if t["contact"]), None)
        if contact_track:
            area = contact_track["bbox"][2] * contact_track["bbox"][3]
            if current is None:
                current = {
                    "frame_start": r["frame"],
                    "frame_end": r["frame"],
                    "centroids": [contact_track["centroid"]],
                    "areas": [area],
                    "pre_area_0": last_area["rat_0"],
                    "pre_area_1": last_area["rat_1"],
                }
            else:
                current["frame_end"] = r["frame"]
                current["centroids"].append(contact_track["centroid"])
                current["areas"].append(area)
        else:
            if current is not None:
                bouts.append(_finalize(current))
                current = None
            for t in r["tracks"]:
                if t["track_id"] in last_area:
                    last_area[t["track_id"]] = t["bbox"][2] * t["bbox"][3]

    if current is not None:
        bouts.append(_finalize(current))
    return bouts
