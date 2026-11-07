from __future__ import annotations

from collections import OrderedDict

import cv2
import numpy as np
from PyQt6.QtCore import QElapsedTimer, QObject, QTimer, pyqtSignal
from PyQt6.QtGui import QImage

SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]
_CACHE_BYTES = 256 * 1024 * 1024


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    h, w, ch = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()


class VideoSource(QObject):
    frame_ready = pyqtSignal(int, object)
    finished = pyqtSignal()
    playing_changed = pyqtSignal(bool)

    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.frame_count <= 0 or self.width <= 0:
            raise IOError(f"Video has no readable frames: {path}")
        self.current_frame = 0
        self.speed = 1.0
        self.loop_range: tuple[int, int] | None = None
        self.loop_enabled = False

        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_bytes = 0
        self._next_decode = 0
        self._clock = QElapsedTimer()
        self._clock_start_frame = 0
        self._timer = QTimer(self)
        self._timer.setInterval(15)
        self._timer.timeout.connect(self._tick)


    def _cache_put(self, idx: int, frame: np.ndarray) -> None:
        if idx in self._cache:
            return
        self._cache[idx] = frame
        self._cache_bytes += frame.nbytes
        while self._cache_bytes > _CACHE_BYTES and len(self._cache) > 1:
            _, old = self._cache.popitem(last=False)
            self._cache_bytes -= old.nbytes

    def read_frame(self, idx: int) -> np.ndarray | None:
        idx = max(0, min(self.frame_count - 1, int(idx)))
        hit = self._cache.get(idx)
        if hit is not None:
            self._cache.move_to_end(idx)
            return hit
        if idx != self._next_decode:
            if 0 < idx - self._next_decode <= 12:
                while self._next_decode < idx:
                    if not self.cap.grab():
                        return None
                    self._next_decode += 1
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                self._next_decode = idx
        ok, frame = self.cap.read()
        if not ok:
            return None
        self._next_decode = idx + 1
        self._cache_put(idx, frame)
        return frame

    def _show(self, idx: int) -> bool:
        frame = self.read_frame(idx)
        if frame is None:
            return False
        self.current_frame = max(0, min(self.frame_count - 1, int(idx)))
        self.frame_ready.emit(self.current_frame, frame)
        return True


    def seek(self, idx: int) -> None:
        self._show(idx)
        if self._timer.isActive():
            self._reset_clock()

    def step(self, delta: int) -> None:
        self._show(self.current_frame + delta)

    def set_speed(self, speed: float) -> None:
        self.speed = float(speed)
        if self._timer.isActive():
            self._reset_clock()

    def set_loop(self, enabled: bool, frame_range: tuple[int, int] | None = None) -> None:
        self.loop_enabled = enabled
        if frame_range is not None:
            self.loop_range = (int(frame_range[0]), int(frame_range[1]))

    def is_playing(self) -> bool:
        return self._timer.isActive()

    def play(self) -> None:
        if self.current_frame >= self.frame_count - 1:
            self._show(0)
        self._reset_clock()
        self._timer.start()
        self.playing_changed.emit(True)

    def pause(self) -> None:
        was = self._timer.isActive()
        self._timer.stop()
        if was:
            self.playing_changed.emit(False)

    def _reset_clock(self) -> None:
        self._clock.start()
        self._clock_start_frame = self.current_frame

    def _tick(self) -> None:
        elapsed = self._clock.elapsed() / 1000.0
        target = self._clock_start_frame + int(elapsed * self.fps * self.speed)
        end = self.frame_count - 1
        if self.loop_enabled and self.loop_range is not None:
            lo, hi = self.loop_range
            if self.current_frame > hi or target > hi:
                self._show(lo)
                self._reset_clock()
                return
            end = hi
        if target > end:
            if self.current_frame < end:
                self._show(end)
            self.pause()
            self.finished.emit()
            return
        if target <= self.current_frame:
            return
        if target - self.current_frame > 1:
            if not self._show(target):
                self.pause()
                self.finished.emit()
            return
        if not self._show(self.current_frame + 1):
            self.pause()
            self.finished.emit()

    def release(self) -> None:
        self._timer.stop()
        self.cap.release()
        self._cache.clear()
        self._cache_bytes = 0
