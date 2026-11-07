from __future__ import annotations

from PyQt6.QtCore import QObject, QThread, pyqtSignal


class FunctionWorker(QObject):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def run(self) -> None:
        def report(done: int, total: int):
            self.progress.emit(int(done), int(total))
            return not self.cancelled

        try:
            result = self.fn(report)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        self.finished.emit(result)


def start_worker(fn, on_finished, on_failed, on_progress=None) -> tuple[QThread, FunctionWorker]:
    thread = QThread()
    worker = FunctionWorker(fn)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(on_finished)
    worker.failed.connect(on_failed)
    if on_progress is not None:
        worker.progress.connect(on_progress)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return thread, worker
