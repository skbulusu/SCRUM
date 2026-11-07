from __future__ import annotations

import sys


def run_gui(video=None, tracks=None, predictions=None, session=None, folder=None, qt_args=None, paths=None) -> int:
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        print(
            "The SCRUM Viewer needs PyQt6. Install it with:\n    pip install \"scrum-behavior[gui]\"",
            file=sys.stderr,
        )
        return 1

    from scrum import __version__
    from scrum.app.main_window import MainWindow

    from PyQt6.QtCore import QEvent

    class ScrumApp(QApplication):
        opener = None

        def event(self, e):
            if e.type() == QEvent.Type.FileOpen and self.opener is not None:
                self.opener([e.file()])
                return True
            return super().event(e)

    app = ScrumApp([sys.argv[0]] + list(qt_args or []))
    app.setApplicationName("SCRUM Viewer")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("SCRUM")
    from PyQt6.QtGui import QIcon

    from scrum.paths import asset_path

    icon = asset_path("icon.png")
    if icon.is_file():
        app.setWindowIcon(QIcon(str(icon)))
    window = MainWindow()
    app.opener = window.open_paths
    window.show()

    if session:
        window.load_session(session)
    else:
        if video:
            window.load_video(video, auto_detect=not (tracks or predictions))
        if tracks:
            window.load_tracks(tracks)
        if predictions:
            window.load_predictions(predictions)
    if folder:
        window._show_batch(folder)
    if paths:
        window.open_paths(list(paths))
    return app.exec()


def main() -> None:
    from scrum.cli import main as cli_main

    sys.exit(cli_main(["gui"] + sys.argv[1:]))


if __name__ == "__main__":
    main()
