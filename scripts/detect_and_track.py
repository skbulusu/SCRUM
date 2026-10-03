from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.blob_tracker import TrackerConfig, track_video, label_contact_bouts

STABLE_IDS = ["rat_0", "rat_1"]
_TEXT_COLORS = {"rat_0": (66, 133, 244), "rat_1": (219, 68, 55)}
_CONTACT_COLOR = (0, 255, 255)
_OUTLINE_COLOR = (20, 255, 57)
_LABEL_BG = (0, 0, 0)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _put_label(frame, text, x, y, color, scale=0.5, thickness=1):
    (tw, th), baseline = cv2.getTextSize(text, _FONT, scale, thickness)
    cv2.rectangle(frame, (x - 3, y - th - 4), (x + tw + 3, y + baseline + 2), _LABEL_BG, -1)
    cv2.putText(frame, text, (x, y), _FONT, scale, color, thickness, cv2.LINE_AA)


def _label_anchor(frame_shape, bbox):
    h_frame, w_frame = frame_shape[:2]
    x, y, w, h = bbox
    above_y = y - 10
    label_y = above_y if above_y - 16 > 0 else y + h + 22
    label_x = int(np.clip(x, 4, max(4, w_frame - 160)))
    return label_x, label_y


def _draw_overlay(frame, tracks):
    contact_track = next((t for t in tracks if t["contact"]), None)
    if contact_track is not None:
        text = "Contact"
        (tw, th), baseline = cv2.getTextSize(text, _FONT, 1.0, 2)
        cx = frame.shape[1] // 2 - tw // 2
        cv2.rectangle(frame, (cx - 10, 8), (cx + tw + 10, 8 + th + baseline + 10), _LABEL_BG, -1)
        cv2.putText(frame, text, (cx, 8 + th + 5), _FONT, 1.0, _CONTACT_COLOR, 2, cv2.LINE_AA)
    else:
        for tid in STABLE_IDS:
            t = next((t for t in tracks if t["track_id"] == tid), None)
            if t is None:
                continue
            x, y = t["display_centroid"]
            lx, ly = _label_anchor(frame.shape, t["bbox"])
            _put_label(frame, f"({x:.0f}, {y:.0f})", lx, ly, _TEXT_COLORS[tid])


def _draw_one_outline(frame, contours):
    if len(contours) <= 1:
        for c in contours:
            cv2.drawContours(frame, [c], -1, _OUTLINE_COLOR, 2, cv2.LINE_AA)
        return

    h_frame, w_frame = frame.shape[:2]
    xs = np.concatenate([c[:, 0, 0] for c in contours])
    ys = np.concatenate([c[:, 0, 1] for c in contours])
    pad = 20
    x0, y0 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
    x1, y1 = min(w_frame, int(xs.max()) + pad), min(h_frame, int(ys.max()) + pad)

    local = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    for c in contours:
        cv2.drawContours(local, [c - [x0, y0]], -1, 255, -1)
    bridge_kernel = np.ones((21, 21), np.uint8)
    local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, bridge_kernel)
    merged, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for mc in merged:
        cv2.drawContours(frame, [mc + [x0, y0]], -1, _OUTLINE_COLOR, 2, cv2.LINE_AA)


def _draw_outlines(frame, tracks):
    for t in tracks:
        _draw_one_outline(frame, t.get("contours", []))
    return frame


def _compose(frame, tracks):
    frame = _draw_outlines(frame, tracks)
    _draw_overlay(frame, tracks)
    return frame


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--export-frames", action="store_true")
    ap.add_argument("--frame-every-n", type=int, default=1)
    ap.add_argument(
        "--diff-threshold", type=int, default=25, help="pixel intensity diff to count as foreground"
    )
    ap.add_argument(
        "--min-blob-area", type=int, default=200, help="ignore detected blobs smaller than this many pixels"
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    if args.export_frames:
        frames_dir.mkdir(exist_ok=True)

    cfg = TrackerConfig(diff_threshold=args.diff_threshold, min_blob_area=args.min_blob_area)
    print("building background model and tracking...")
    results = track_video(args.video, cfg)

    rows = []
    for r in results:
        for t in r["tracks"]:
            x, y, w, h = t["bbox"]
            dc = t.get("display_centroid")
            rows.append(
                {
                    "frame": r["frame"],
                    "track_id": t["track_id"],
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "centroid_x": t["centroid"][0],
                    "centroid_y": t["centroid"][1],
                    "display_x": dc[0] if dc else None,
                    "display_y": dc[1] if dc else None,
                    "contact": t["contact"],
                }
            )
    tracks_df = pd.DataFrame(rows)
    tracks_df.to_csv(out_dir / "tracks.csv", index=False)
    n_contact = int(tracks_df[tracks_df.contact].frame.nunique()) if len(tracks_df) else 0
    n_frames = int(tracks_df.frame.nunique()) if len(tracks_df) else 0
    print(f"tracked {n_frames} frames, {n_contact} flagged as contact/overlap -> {out_dir / 'tracks.csv'}")

    bouts = label_contact_bouts(results)
    pd.DataFrame(bouts).to_csv(out_dir / "contact_bouts.csv", index=False)
    print(f"{len(bouts)} contact bout(s), heuristic behavior guesses -> {out_dir / 'contact_bouts.csv'}")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    tracks_by_frame = {r["frame"]: r["tracks"] for r in results}
    frame_idx = 0
    written_images = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        composed = _compose(frame, tracks_by_frame.get(frame_idx, []))
        writer.write(composed)
        if args.export_frames and frame_idx % args.frame_every_n == 0:
            cv2.imwrite(str(frames_dir / f"frame_{frame_idx:06d}.jpg"), composed)
            written_images += 1
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"annotated video -> {out_dir / 'annotated.mp4'}")
    if args.export_frames:
        print(f"{written_images} annotated frame image(s) -> {frames_dir}")


if __name__ == "__main__":
    main()
