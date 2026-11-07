from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from scrum.app.data import behavior_bgr

FONT = cv2.FONT_HERSHEY_SIMPLEX

OUTLINE_GREEN = (89, 199, 52)
ANIMAL_COLORS = [(10, 159, 255), (255, 210, 100), (88, 209, 48), (242, 90, 191)]
MERGED_AMBER = (10, 159, 255)
_WHITE_ISH = 235


def animal_color(track_id: str, animals: list[str]) -> tuple[int, int, int]:
    if track_id in animals:
        return ANIMAL_COLORS[animals.index(track_id) % len(ANIMAL_COLORS)]
    return OUTLINE_GREEN


def _qt_available() -> bool:
    try:
        from PyQt6.QtGui import QGuiApplication
    except ImportError:
        return False
    return QGuiApplication.instance() is not None


def _label_font(scale: float):
    from PyQt6.QtGui import QFont, QGuiApplication

    font = QFont(QGuiApplication.font())
    font.setPixelSize(max(10, round(20 * scale)))
    font.setWeight(QFont.Weight.DemiBold)
    return font


def _has_dot(color) -> bool:
    return min(color) < _WHITE_ISH


def label_width(text: str, scale: float = 0.6, color=(255, 255, 255)) -> int:
    if not _qt_available():
        return cv2.getTextSize(text, FONT, scale, 1)[0][0] + 8
    from PyQt6.QtGui import QFontMetrics

    px = max(10, round(20 * scale))
    tw = QFontMetrics(_label_font(scale)).horizontalAdvance(text)
    return int(np.ceil(px * 1.2 + tw + (px * 0.9 if _has_dot(color) else 0)))


def put_label(frame: np.ndarray, text: str, org: tuple[int, int], color=(255, 255, 255), scale: float = 0.6,
              thickness: int = 1, bg=(0, 0, 0), alpha: float = 0.6) -> tuple[int, int, int, int]:
    if not _qt_available():
        return _put_label_cv(frame, text, org, color, scale, thickness, bg, alpha)
    from PyQt6.QtCore import QRectF, Qt
    from PyQt6.QtGui import QColor, QFontMetrics, QImage, QPainter

    font = _label_font(scale)
    px = font.pixelSize()
    fm = QFontMetrics(font)
    dot = _has_dot(color)
    pad, dot_d, gap = px * 0.6, px * 0.5, px * 0.4
    w = label_width(text, scale, color)
    h = int(np.ceil(px * 1.75))
    img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    q = QPainter(img)
    q.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
    fill = QColor(bg[2] or 28, bg[1] or 28, bg[0] or 30)
    fill.setAlphaF(float(np.clip(alpha + 0.1, 0.0, 0.85)))
    q.setPen(Qt.PenStyle.NoPen)
    q.setBrush(fill)
    q.drawRoundedRect(QRectF(0, 0, w, h), h / 2, h / 2)
    x = pad
    if dot:
        q.setBrush(QColor(color[2], color[1], color[0]))
        q.drawEllipse(QRectF(x, (h - dot_d) / 2, dot_d, dot_d))
        x += dot_d + gap
    q.setPen(QColor(255, 255, 255))
    q.setFont(font)
    q.drawText(QRectF(x, 0, w - x, h), int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), text)
    q.end()

    stride = img.bytesPerLine()
    rgba = np.frombuffer(img.constBits().asarray(stride * h), dtype=np.uint8).reshape(h, stride // 4, 4)[:, :w]
    left = int(org[0]) - int(pad * 0.5)
    top = int(org[1]) + int(fm.descent()) - h + int(h * 0.12)
    fh, fw = frame.shape[:2]
    x0, y0, x1, y1 = max(0, left), max(0, top), min(fw, left + w), min(fh, top + h)
    if x1 > x0 and y1 > y0:
        patch = rgba[y0 - top : y1 - top, x0 - left : x1 - left]
        a = patch[..., 3:4].astype(np.float32) / 255.0
        roi = frame[y0:y1, x0:x1]
        roi[:] = np.clip(patch[..., :3].astype(np.float32) + roi.astype(np.float32) * (1.0 - a), 0, 255).astype(np.uint8)
    return (x0, y0, x1, y1)


def _put_label_cv(frame, text, org, color, scale, thickness, bg, alpha):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = int(org[0]), int(org[1])
    x0, y0 = max(0, x - 4), max(0, y - th - 4)
    x1, y1 = min(frame.shape[1], x + tw + 4), min(frame.shape[0], y + base + 2)
    if x1 > x0 and y1 > y0:
        roi = frame[y0:y1, x0:x1]
        cv2.addWeighted(np.full_like(roi, bg), alpha, roi, 1 - alpha, 0, dst=roi)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)
    return (x0, y0, x1, y1)


def draw_contours(frame: np.ndarray, contours: list, color=OUTLINE_GREEN, thickness: int = 2) -> None:
    contours = [c for c in contours if c is not None and len(c)]
    if not contours:
        return
    if len(contours) == 1:
        cv2.drawContours(frame, contours, -1, color, thickness, cv2.LINE_AA)
        return
    h_frame, w_frame = frame.shape[:2]
    xs = np.concatenate([c[:, 0, 0] for c in contours])
    ys = np.concatenate([c[:, 0, 1] for c in contours])
    pad = 20
    x0, y0 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
    x1, y1 = min(w_frame, int(xs.max()) + pad), min(h_frame, int(ys.max()) + pad)
    if x1 <= x0 or y1 <= y0:
        return
    local = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    for c in contours:
        cv2.drawContours(local, [c - [x0, y0]], -1, 255, -1)
    local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    merged, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for mc in merged:
        cv2.drawContours(frame, [mc + [x0, y0]], -1, color, thickness, cv2.LINE_AA)


def draw_trail(frame: np.ndarray, pts: np.ndarray, color, thickness: int = 2, chunks: int = 6) -> None:
    ok = ~np.isnan(pts).any(axis=1)
    if ok.sum() < 2:
        return
    n = len(pts)
    bounds = np.linspace(0, n - 1, chunks + 1).astype(int)
    for c in range(chunks):
        seg = pts[bounds[c] : bounds[c + 1] + 1]
        keep = ~np.isnan(seg).any(axis=1)
        if keep.sum() < 2:
            continue
        seg_i = seg.round().astype(np.int32)
        x0, y0 = max(0, seg_i[keep][:, 0].min() - 4), max(0, seg_i[keep][:, 1].min() - 4)
        x1 = min(frame.shape[1], seg_i[keep][:, 0].max() + 5)
        y1 = min(frame.shape[0], seg_i[keep][:, 1].max() + 5)
        if x1 <= x0 or y1 <= y0:
            continue
        roi = frame[y0:y1, x0:x1]
        layer = roi.copy()
        for i in range(len(seg_i) - 1):
            if keep[i] and keep[i + 1]:
                cv2.line(layer, tuple(seg_i[i] - [x0, y0]), tuple(seg_i[i + 1] - [x0, y0]), color, thickness, cv2.LINE_AA)
        alpha = 0.25 + 0.75 * (c + 1) / chunks
        cv2.addWeighted(layer, alpha, roi, 1 - alpha, 0, dst=roi)


def draw_zone(frame: np.ndarray, name: str, polygon, color=(180, 105, 255), label: bool = True) -> None:
    pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
    if len(pts) < 2:
        return
    cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
    if label:
        x, y = int(pts[:, 0, 0].min()), int(pts[:, 0, 1].min())
        put_label(frame, name, (x + 4, y + 18), color=color, scale=0.55)


def draw_distance(frame: np.ndarray, a: tuple[float, float], b: tuple[float, float], text: str) -> None:
    p, q = (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))
    cv2.line(frame, p, q, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.line(frame, p, q, (255, 255, 255), 2, cv2.LINE_AA)
    w = label_width(text, 0.55)
    mid = ((p[0] + q[0]) // 2 - w // 2, (p[1] + q[1]) // 2 + 34)
    put_label(frame, text, mid, scale=0.55)


def draw_magnifier(frame: np.ndarray, source: np.ndarray, center: tuple[float, float], span: float, zoom: float = 3.0) -> None:
    h, w = frame.shape[:2]
    size = int(min(max(min(h, w) * 0.42, 200), min(h, w) * 0.6))
    need = span + 90


    eff_zoom = max(2.0, min(zoom, size / need))
    side = int(min(min(h, w), size / eff_zoom))
    x0 = int(np.clip(center[0] - side / 2, 0, w - side))
    y0 = int(np.clip(center[1] - side / 2, 0, h - side))
    crop = source[y0 : y0 + side, x0 : x0 + side]
    if crop.size == 0:
        return
    inset = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    margin = 8
    corners = [(margin, margin), (w - size - margin, margin), (margin, h - size - margin), (w - size - margin, h - size - margin)]
    ix, iy = max(corners, key=lambda c: np.hypot(c[0] + size / 2 - center[0], c[1] + size / 2 - center[1]))
    frame[iy : iy + size, ix : ix + size] = inset
    cv2.rectangle(frame, (ix - 1, iy - 1), (ix + size, iy + size), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (x0, y0), (x0 + side, y0 + side), (255, 255, 255), 1, cv2.LINE_AA)
    put_label(frame, f"{eff_zoom:.1f}x", (ix + 6, iy + 20), scale=0.5)


def draw_outlines(frame: np.ndarray, frame_outlines: list[list] | None) -> np.ndarray:
    if not frame_outlines:
        return frame
    for contours in frame_outlines:
        draw_contours(frame, contours)
    return frame


def draw_overlay(frame: np.ndarray, frame_tracks: pd.DataFrame | None, behavior: str | None) -> np.ndarray:
    if frame_tracks is not None and not frame_tracks.empty:
        ids = sorted(frame_tracks["track_id"].unique())
        for _, pt in frame_tracks.iterrows():
            color = animal_color(pt["track_id"], ids)
            x, y = int(pt["x"]), int(pt["y"])
            cv2.circle(frame, (x, y), 5, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)
    if behavior is not None:
        text = f"prediction: {behavior}"
        color = behavior_bgr(behavior)
        put_label(frame, text, (frame.shape[1] // 2 - label_width(text, 0.8, color) // 2, 40), color=color, scale=0.8, thickness=2, alpha=0.8)
    return frame
