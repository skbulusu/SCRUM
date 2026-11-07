from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


class CVKalman:
    def __init__(self, xy: tuple[float, float], process_var: float = 4.0, meas_var: float = 9.0):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0], dtype=float)
        self.P = np.diag([meas_var, meas_var, 100.0, 100.0]).astype(float)
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.q = process_var
        self.default_r = meas_var

    @property
    def position(self) -> tuple[float, float]:
        return (float(self.x[0]), float(self.x[1]))

    def predict(self, velocity_decay: float = 1.0) -> tuple[float, float]:
        self.x = self.F @ self.x
        self.x[2:] *= velocity_decay
        Q = np.diag([self.q * 0.25, self.q * 0.25, self.q, self.q])
        self.P = self.F @ self.P @ self.F.T + Q
        return self.position

    def update(self, xy: tuple[float, float], meas_var: float | None = None) -> None:
        r = self.default_r if meas_var is None else meas_var
        R = np.eye(2) * r
        z = np.asarray(xy, dtype=float)
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P


class IdentityKalman:
    def __init__(
        self,
        ids: list[str],
        process_var: float = 4.0,
        meas_var: float = 9.0,
        merged_meas_var: float = 100.0,
        coast_decay: float = 0.97,
    ):
        self.ids = list(ids)
        self.process_var = process_var
        self.meas_var = meas_var
        self.merged_meas_var = merged_meas_var
        self.coast_decay = coast_decay
        self.filters: dict[str, CVKalman] = {}
        self.frames_since_direct: dict[str, int] = {tid: 0 for tid in ids}
        self._measured_now: set[str] = set()

    def begin_frame(self) -> None:
        self._measured_now = set()
        for tid, f in self.filters.items():
            decay = self.coast_decay if self.frames_since_direct[tid] > 0 else 1.0
            f.predict(velocity_decay=decay)

    def end_frame(self) -> None:
        for tid in self.ids:
            if tid in self._measured_now:
                self.frames_since_direct[tid] = 0
            else:
                self.frames_since_direct[tid] += 1

    def update(self, tid: str, xy: tuple[float, float], merged: bool = False) -> None:
        r = self.merged_meas_var if merged else self.meas_var
        if tid not in self.filters:
            self.filters[tid] = CVKalman(xy, self.process_var, self.meas_var)
        else:
            self.filters[tid].update(xy, r)
        if not merged:
            self._measured_now.add(tid)

    def predicted(self) -> dict[str, tuple[float, float]]:
        return {tid: f.position for tid, f in self.filters.items()}

    def assign(self, points: list[tuple[float, float]]) -> list[str | None]:
        out: list[str | None] = [None] * len(points)
        pred = self.predicted()
        if len(pred) < len(self.ids) or not points:
            order = np.argsort([p[0] for p in points])
            for rank, i in enumerate(order):
                out[int(i)] = self.ids[rank] if rank < len(self.ids) else None
            return out
        cost = np.zeros((len(self.ids), len(points)))
        for i, tid in enumerate(self.ids):
            px, py = pred[tid]
            for j, p in enumerate(points):
                cost[i, j] = float(np.hypot(px - p[0], py - p[1]))
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            out[int(c)] = self.ids[int(r)]
        return out

    def assign_with_margin(self, points: list[tuple[float, float]]) -> tuple[list[str | None], float]:
        ids = self.assign(points)
        pred = self.predicted()
        if len(points) != 2 or len(self.ids) != 2 or len(pred) < 2 or None in ids:
            return ids, 0.0
        a, b = self.ids
        d = lambda tid, pt: float(np.hypot(pred[tid][0] - pt[0], pred[tid][1] - pt[1]))
        straight = d(a, points[0]) + d(b, points[1])
        swapped = d(a, points[1]) + d(b, points[0])
        chosen = straight if ids[0] == a else swapped
        return ids, abs(straight - swapped) if chosen == min(straight, swapped) else 0.0

    def should_swap(self, p0: tuple[float, float], p1: tuple[float, float], swap_ratio: float = 0.7,
                    min_cost: float = 20.0) -> bool:
        a, b = self.ids[0], self.ids[1]
        if a not in self.filters or b not in self.filters:
            return False
        if max(self.frames_since_direct[a], self.frames_since_direct[b]) < 1:
            return False
        pa, pb = self.filters[a].position, self.filters[b].position
        straight = np.hypot(pa[0] - p0[0], pa[1] - p0[1]) + np.hypot(pb[0] - p1[0], pb[1] - p1[1])
        swapped = np.hypot(pa[0] - p1[0], pa[1] - p1[1]) + np.hypot(pb[0] - p0[0], pb[1] - p0[1])
        return bool(straight > min_cost and swapped < swap_ratio * straight)
