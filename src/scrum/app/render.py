from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import cv2
import numpy as np

from scrum.app import overlay
from scrum.app.data import behavior_bgr
from scrum.app.overlay import MERGED_AMBER, OUTLINE_GREEN, animal_color, put_label
from scrum.app.project import Project

MERGED_ID = "rat_0+rat_1"


@dataclass
class OverlayOptions:
    outlines: bool = True
    split_outlines: bool = True
    centroids: bool = True
    keypoints: bool = True
    ids: bool = True
    trails: bool = True
    distance: bool = True
    zones: bool = True
    magnifier: bool = False
    behavior_label: bool = True
    confidence: bool = True
    merged_tag: bool = True
    trail_seconds: float = 3.0
    magnifier_zoom: float = 3.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OverlayOptions":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


TOGGLES = [
    ("outlines", "Body outlines"),
    ("split_outlines", "Per-animal outlines when touching"),
    ("centroids", "Centroids"),
    ("keypoints", "Keypoints"),
    ("ids", "Animal IDs"),
    ("trails", "Motion trails"),
    ("distance", "Distance line and readout"),
    ("zones", "Zones"),
    ("magnifier", "Magnifier on the pair"),
    ("behavior_label", "Behavior label"),
    ("confidence", "Confidence in label"),
    ("merged_tag", "Merged-blob tag"),
]

_SKELETON = [("nose", "neck"), ("neck", "tailbase"), ("left_ear", "nose"), ("right_ear", "nose"),
             ("neck", "left_hip"), ("neck", "right_hip")]


class FrameRenderer:
    def __init__(self, project: Project, options: OverlayOptions | None = None):
        self.project = project
        self.options = options or OverlayOptions()

    def render(self, frame: np.ndarray, idx: int) -> np.ndarray:
        p, o = self.project, self.options
        out = frame.copy()
        animals = p.animals()

        if o.zones:
            for name, poly in p.zones.items():
                overlay.draw_zone(out, name, poly)

        if o.outlines:
            self._draw_outlines(out, idx, animals)

        if o.trails and p.positions:
            n = max(2, int(o.trail_seconds * p.fps))
            lo = max(0, idx - n)
            for a, pos in p.positions.items():
                overlay.draw_trail(out, pos[lo : idx + 1], animal_color(a, animals))

        self._draw_points(out, idx, animals)

        if o.distance and len(p.positions) >= 2 and idx < len(p.distance):
            ids = sorted(p.positions)
            a, b = p.positions[ids[0]][idx], p.positions[ids[1]][idx]
            d = p.distance[idx]
            if not (np.isnan(a).any() or np.isnan(b).any() or np.isnan(d)):
                overlay.draw_distance(out, a, b, f"{d:.1f} {p.units}")

        if o.magnifier and p.positions:
            pts = [pos[idx] for pos in p.positions.values() if idx < len(pos) and not np.isnan(pos[idx]).any()]
            if pts:
                c = np.mean(pts, axis=0)
                span = max((float(np.hypot(*(q - r))) for q in pts for r in pts), default=0.0)
                overlay.draw_magnifier(out, out.copy(), (c[0], c[1]), span, o.magnifier_zoom)

        if o.behavior_label:
            self._draw_behavior(out, idx)
        if o.merged_tag and idx < len(p.merged) and p.merged[idx]:
            put_label(out, "merged blob", (10, out.shape[0] - 12), color=MERGED_AMBER, scale=0.55)
        return out


    def _draw_outlines(self, out: np.ndarray, idx: int, animals: list[str]) -> None:
        entries = self.project.outlines.get(idx, [])
        use_split = self.options.split_outlines and any(e.get("split") for e in entries)
        for e in entries:
            if e.get("split"):
                if use_split:
                    overlay.draw_contours(out, e["contours"], animal_color(e["track_id"], animals))
            elif e.get("merged") and use_split:
                continue
            else:
                overlay.draw_contours(out, e["contours"], OUTLINE_GREEN)

    def _draw_points(self, out: np.ndarray, idx: int, animals: list[str]) -> None:
        o = self.options
        rows = self.project.tracks_lookup.get(idx, [])
        by_animal: dict[str, dict[str, tuple[int, int]]] = {}
        for tid, kp, x, y in rows:
            by_animal.setdefault(tid, {})[kp] = (int(round(x)), int(round(y)))
        for tid, kps in by_animal.items():
            color = animal_color(tid, animals) if tid != MERGED_ID else MERGED_AMBER
            if o.keypoints and any(k != "centroid" for k in kps):
                for a, b in _SKELETON:
                    if a in kps and b in kps:
                        cv2.line(out, kps[a], kps[b], color, 1, cv2.LINE_AA)
                for k, pt in kps.items():
                    if k != "centroid":
                        cv2.circle(out, pt, 3, color, -1, cv2.LINE_AA)
            if o.centroids and "centroid" in kps:
                cv2.circle(out, kps["centroid"], 5, color, -1, cv2.LINE_AA)
                cv2.circle(out, kps["centroid"], 5, (255, 255, 255), 1, cv2.LINE_AA)
            if o.ids:
                anchor = kps.get("centroid") or kps.get("nose") or next(iter(kps.values()))
                put_label(out, tid, (anchor[0] + 8, anchor[1] - 8), color=color, scale=0.5)

    def _draw_behavior(self, out: np.ndarray, idx: int) -> None:
        p, o = self.project, self.options
        label = p.label_at(idx)
        if label is None:
            return
        text = label
        if o.confidence and idx < len(p.conf) and not np.isnan(p.conf[idx]) and p.manual[idx] == "":
            text += f"  {p.conf[idx]:.2f}"
        if p.manual[idx] != "":
            pred = p.pred[idx] or "none"
            text += "  (manual)" if pred == label else f"  (manual, model: {pred})"
        scale = float(np.clip(0.62 * out.shape[1] / 640.0, 0.5, 1.4))
        color = behavior_bgr(label)
        tw = overlay.label_width(text, scale, color)
        put_label(out, text, (out.shape[1] // 2 - tw // 2, int(22 + 22 * scale)), color=color, scale=scale, thickness=2, alpha=0.8)
