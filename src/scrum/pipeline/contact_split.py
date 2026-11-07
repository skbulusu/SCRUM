from __future__ import annotations

import cv2
import numpy as np


def fit_ellipse_contour(contour: np.ndarray) -> tuple[float, float, float] | None:
    if contour is None or len(contour) < 5:
        return None
    pts = contour.reshape(-1, 2).astype(np.float32).reshape(-1, 1, 2)
    (_cx, _cy), (a1, a2), angle = cv2.fitEllipse(pts)
    major, minor = max(a1, a2), min(a1, a2)
    if minor <= 0:
        return None
    return (float(major), float(minor), float(angle % 180.0))


def _part_from_mask(part_mask: np.ndarray, offset: tuple[int, int]) -> dict | None:
    contours, _ = cv2.findContours(part_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(part_mask.sum())
    if area <= 0:
        return None
    m = cv2.moments(part_mask.astype(np.uint8), binaryImage=True)
    mass = (m["m10"] / m["m00"] + offset[0], m["m01"] / m["m00"] + offset[1]) if m["m00"] else None
    x, y, w, h = cv2.boundingRect(contour)
    bbox = (x + offset[0], y + offset[1], w, h)
    contour_full = contour + np.array([[offset[0], offset[1]]], dtype=contour.dtype)
    return {
        "bbox": bbox,
        "centroid": (bbox[0] + w / 2.0, bbox[1] + h / 2.0),
        "display_centroid": mass if mass is not None else (bbox[0] + w / 2.0, bbox[1] + h / 2.0),
        "area": area,
        "contours": [contour_full],
        "ellipse": fit_ellipse_contour(contour_full),
    }


def _neck_ratio(dist: np.ndarray, p1: tuple[int, int], p2: tuple[int, int]) -> float:
    n = int(max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1]))) + 1
    xs = np.linspace(p1[0], p2[0], n).round().astype(int)
    ys = np.linspace(p1[1], p2[1], n).round().astype(int)
    xs = np.clip(xs, 0, dist.shape[1] - 1)
    ys = np.clip(ys, 0, dist.shape[0] - 1)
    line = dist[ys, xs]
    lower_peak = min(dist[p1[1], p1[0]], dist[p2[1], p2[0]])
    if lower_peak <= 0:
        return 1.0
    return float(np.clip(line.min() / lower_peak, 0.0, 1.0))


def _find_peaks(dist: np.ndarray, min_frac: float = 0.35) -> list[tuple[int, int, float]]:
    dmax = float(dist.max())
    if dmax <= 0:
        return []
    k = max(5, int(dmax * 1.5) | 1)
    local_max = (dist >= cv2.dilate(dist, np.ones((k, k), np.uint8)) - 1e-6) & (dist > min_frac * dmax)
    n, labels = cv2.connectedComponents(local_max.astype(np.uint8))
    peaks = []
    for lab in range(1, n):
        ys, xs = np.nonzero(labels == lab)
        j = int(np.argmax(dist[ys, xs]))
        peaks.append((int(xs[j]), int(ys[j]), float(dist[ys[j], xs[j]])))
    peaks.sort(key=lambda p: -p[2])
    return peaks


def _neck_cut_two(mask: np.ndarray, dist: np.ndarray, p1, p2) -> tuple[np.ndarray, np.ndarray]:
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    n = int(max(abs(x2 - x1), abs(y2 - y1))) + 1
    xs = np.linspace(x1, x2, n)
    ys = np.linspace(y1, y2, n)
    line = dist[np.clip(ys.round().astype(int), 0, dist.shape[0] - 1), np.clip(xs.round().astype(int), 0, dist.shape[1] - 1)]
    narrow = np.nonzero(line <= line.min() + 0.75)[0]
    k = int(np.median(narrow))
    nx, ny = xs[k], ys[k]
    ux, uy = x2 - x1, y2 - y1
    norm = float(np.hypot(ux, uy)) or 1.0
    ux, uy = ux / norm, uy / norm
    yy, xx = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
    side = (xx - nx) * ux + (yy - ny) * uy
    fg = mask > 0
    return fg & (side < 0), fg & (side >= 0)


def _kmeans_two(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _compact, labels, _centers = cv2.kmeans(pts, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    a = np.zeros(mask.shape, bool)
    b = np.zeros(mask.shape, bool)
    a[ys[labels == 0], xs[labels == 0]] = True
    b[ys[labels == 1], xs[labels == 1]] = True
    return a, b


def split_blob_mask(
    mask: np.ndarray,
    offset: tuple[int, int] = (0, 0),
    min_part_area: int = 80,
    max_neck_ratio: float = 0.85,
) -> list[dict] | None:
    mask = (mask > 0).astype(np.uint8)
    if int(mask.sum()) < 2 * min_part_area:
        return None

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    peaks = _find_peaks(dist)

    halves = None
    method = "kmeans"
    confidence = 0.2
    if len(peaks) >= 2:
        p1 = peaks[0]
        for cand in peaks[1:]:
            separation = float(np.hypot(cand[0] - p1[0], cand[1] - p1[1]))
            if separation >= 0.9 * (p1[2] + cand[2]) * 0.6:
                p2 = cand
                ratio = _neck_ratio(dist, (p1[0], p1[1]), (p2[0], p2[1]))
                if ratio <= max_neck_ratio:
                    halves = _neck_cut_two(mask, dist, p1, p2)
                    method = "neck_cut"
                    confidence = float(np.clip(1.0 - ratio, 0.05, 1.0))
                break

    if halves is None or min(int(halves[0].sum()), int(halves[1].sum())) < min_part_area:
        halves = _kmeans_two(mask)
        method = "kmeans"
        confidence = 0.2
    if halves is None:
        return None

    parts = []
    for half in halves:
        if int(half.sum()) < min_part_area:
            return None
        part = _part_from_mask(half, offset)
        if part is None:
            return None
        part["method"] = method
        part["confidence"] = confidence
        parts.append(part)
    parts.sort(key=lambda p: p["centroid"][0])
    return parts


def split_contact_blob(
    gray: np.ndarray, background: np.ndarray, bbox: tuple, diff_threshold: int, morph_kernel: int,
    min_part_area: int = 80, pad: int = 10,
) -> list[dict] | None:
    H, W = gray.shape[:2]
    x, y, w, h = bbox
    x0, y0 = max(0, int(x) - pad), max(0, int(y) - pad)
    x1, y1 = min(W, int(x + w) + pad), min(H, int(y + h) + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    diff = cv2.absdiff(gray[y0:y1, x0:x1], background[y0:y1, x0:x1])
    _, m = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    k = np.ones((morph_kernel, morph_kernel), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return split_blob_mask(m, (x0, y0), min_part_area=min_part_area)
