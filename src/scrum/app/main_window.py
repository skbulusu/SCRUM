from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QSettings, Qt, QTimer
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from scrum.app import export, shortcuts
from scrum.app.analysis import Calibration
from scrum.app.banner import StatusBanner
from scrum.app.batch import VIDEO_EXTENSIONS, BatchWindow
from scrum.app.data import csv_kind, load_predictions, load_tracks
from scrum.app.inference import run_calms21_keypoint_detection, run_scrum_detection_full
from scrum.app.panels import (
    AnnotatePanel,
    BookmarksPanel,
    HeatmapDialog,
    ReviewPanel,
    ShortcutsDialog,
    SummaryPanel,
    TransitionDialog,
    ZonesPanel,
)
from scrum.app.project import Project
from scrum.app.render import TOGGLES, FrameRenderer, OverlayOptions
from scrum.app.results import DetectionResult, cache_path_for, find_cache, results_dir_for
from scrum.app.theme import CHOICES, apply_theme
from scrum.app.timeline import Timeline, fmt_time
from scrum.app.traces import TraceStrip
from scrum.app.video_player import SPEEDS, VideoSource, bgr_to_qimage
from scrum.app.video_view import VideoView

SESSION_EXT = ".scrum"
MAX_RECENT = 10
CORNER_NAMES = ["top-left", "top-right", "bottom-right", "bottom-left"]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SCRUM Viewer")
        self.resize(1360, 900)
        self.setAcceptDrops(True)

        self.settings = QSettings("SCRUM", "Viewer")
        self.theme_name = self.settings.value("theme", "system")
        if self.theme_name not in CHOICES:
            self.theme_name = "system"
        self.theme = apply_theme(QApplication.instance(), self.theme_name)

        self.project = Project()
        try:
            opts = OverlayOptions.from_dict(json.loads(self.settings.value("overlays", "{}")))
        except (ValueError, TypeError):
            opts = OverlayOptions()
        self.renderer = FrameRenderer(self.project, opts)

        self.video: VideoSource | None = None
        self.mode = "scrum"
        self.use_split = True
        self.session_path: str | None = None
        self._dirty = False
        self._rec: tuple[str, int] | None = None
        self._pick: dict | None = None
        self._tasks: list = []
        self._detect_token = 0
        self._detect_worker = None
        self._behavior_actions: list[QAction] = []
        self._batch: BatchWindow | None = None
        self._playing_before_pick = False

        self._build_central()
        self._build_docks()
        self._build_actions()
        self._build_menus()
        self._wire()
        self._set_theme(self.theme_name, save=False)
        self._rebuild_behavior_keys()
        self._update_enabled()

        geo, state = self.settings.value("geometry"), self.settings.value("windowState")
        if geo:
            self.restoreGeometry(geo)
        if state:
            self.restoreState(state)


    def _build_central(self) -> None:
        self.banner = StatusBanner(self.theme)
        self.view = VideoView(self.theme)

        self.play_button = QPushButton("Play")
        self.step_back_button = QPushButton("<")
        self.step_fwd_button = QPushButton(">")
        for b in (self.play_button, self.step_back_button, self.step_fwd_button):
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.step_back_button.setFixedWidth(34)
        self.step_fwd_button.setFixedWidth(34)
        self.speed_combo = QComboBox()
        for s in SPEEDS:
            self.speed_combo.addItem(f"{s:g}x", s)
        self.speed_combo.setCurrentIndex(SPEEDS.index(1.0))
        self.speed_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.loop_check = QCheckBox("Loop selection")
        self.loop_check.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.frame_label = QLabel("frame: -")
        self.frame_label.setMinimumWidth(200)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("SCRUM (raw video)", "scrum")
        self.mode_combo.addItem("CalMS21 / pose tracks", "calms21")
        self.mode_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.model_label = QLabel("model: -")
        self.model_label.setObjectName("muted")

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Mode:"))
        controls.addWidget(self.mode_combo)
        controls.addSpacing(10)
        controls.addWidget(self.step_back_button)
        controls.addWidget(self.play_button)
        controls.addWidget(self.step_fwd_button)
        controls.addWidget(QLabel("Speed"))
        controls.addWidget(self.speed_combo)
        controls.addWidget(self.loop_check)
        controls.addSpacing(10)
        controls.addWidget(self.frame_label)
        controls.addStretch(1)
        controls.addWidget(self.model_label)

        self.timeline = Timeline(self.project, self.theme)
        self.traces = TraceStrip(self.project, self.theme)

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(self.banner)
        layout.addWidget(self.view, stretch=1)
        layout.addLayout(controls)
        layout.addWidget(self.timeline)
        layout.addWidget(self.traces)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

    def _build_docks(self) -> None:
        self.summary_panel = SummaryPanel(self.project)
        self.annotate_panel = AnnotatePanel(self.project)
        self.review_panel = ReviewPanel(self.project)
        self.bookmarks_panel = BookmarksPanel(self.project, lambda: self.video.current_frame if self.video else 0)
        self.zones_panel = ZonesPanel(self.project)
        self.docks: dict[str, QDockWidget] = {}
        order = [("Summary", self.summary_panel), ("Annotate", self.annotate_panel), ("Review", self.review_panel),
                 ("Bookmarks", self.bookmarks_panel), ("Zones", self.zones_panel)]
        first = None
        for title, widget in order:
            dock = QDockWidget(title, self)
            dock.setObjectName("dock_" + title.replace(" ", "_"))
            dock.setWidget(widget)
            widget.setMinimumWidth(400)
            dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetClosable)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
            if first is None:
                first = dock
            else:
                self.tabifyDockWidget(first, dock)
            self.docks[title] = dock
        first.raise_()
        self.setDocumentMode(True)
        self.resizeDocks([first], [420], Qt.Orientation.Horizontal)
        self.setTabPosition(Qt.DockWidgetArea.RightDockWidgetArea, QTabWidget.TabPosition.North)


    def _act(self, text: str, handler, sid: str | None = None, checkable: bool = False) -> QAction:
        a = QAction(text, self)
        a.setCheckable(checkable)
        if sid:
            key = next((k for i, k, _c, _d in shortcuts.SHORTCUTS if i == sid), None)
            if key:
                a.setShortcut(QKeySequence(key))
                a.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        if sid == "delete_bout":
            a.setShortcuts([QKeySequence("Delete"), QKeySequence("Backspace")])
        (a.toggled if checkable else a.triggered).connect(handler)
        self.addAction(a)
        return a

    def _build_actions(self) -> None:
        A = self._act
        self.a_open_video = A("Open Video...", lambda: self._open_video_dialog(), "open_video")
        self.a_open_tracks = A("Open Tracks / Pose...", lambda: self._open_tracks_dialog())
        self.a_open_preds = A("Open Predictions CSV...", lambda: self._open_predictions_dialog())
        self.a_open_session = A("Open Session...", lambda: self._open_session_dialog())
        self.a_save_session = A("Save Session", lambda: self.save_session(), "save_session")
        self.a_save_session_as = A("Save Session As...", lambda: self.save_session(ask=True))
        self.a_quit = A("Quit", self.close)
        self.a_quit.setMenuRole(QAction.MenuRole.QuitRole)

        self.a_export_clip = A("Annotated clip (MP4)...", lambda: self._export_clip(gif=False))
        self.a_export_gif = A("Annotated GIF...", lambda: self._export_clip(gif=True))
        self.a_export_labels = A("Frame labels CSV...", lambda: self._export_table("labels"))
        self.a_export_bouts = A("Bouts CSV...", lambda: self._export_table("bouts"))
        self.a_export_summary = A("Summary CSV...", lambda: self._export_table("summary"))
        self.a_export_boris = A("BORIS-style events CSV...", lambda: self._export_table("boris"))
        self.a_export_pdf = A("PDF report...", lambda: self._export_pdf())

        self.a_undo = A("Undo", lambda: self._undo(), "undo")
        self.a_redo = A("Redo", lambda: self._redo(), "redo")
        self.a_start_manual = A("Start manual labels from predictions", lambda: self.project.start_manual_from_predictions())
        self.a_clear_manual = A("Discard all manual edits", lambda: self._clear_manual())
        self.a_delete_bout = A("Delete selected bout", lambda: self._delete_selected_bout(), "delete_bout")
        self.a_split = A("Split bout at playhead", lambda: self._split_at_playhead(), "split_bout")
        self.a_merge = A("Merge selected bout with next", lambda: self._merge_selected(), "merge_bout")
        self.a_paint = A("Paint tool", lambda on: self._set_paint(on), "paint", checkable=True)
        self.a_bookmark = A("Bookmark this frame", lambda: self._bookmark_here(), "bookmark")

        self.a_zoom_in = A("Zoom timeline in", lambda: self.timeline.zoom_in(), "zoom_in")
        self.a_zoom_out = A("Zoom timeline out", lambda: self.timeline.zoom_out(), "zoom_out")
        self.a_zoom_fit = A("Fit timeline to video", lambda: self.timeline.fit(), "zoom_fit")
        self.appearance_group = QActionGroup(self)
        self.appearance_actions: dict[str, QAction] = {}
        for choice, label in (("system", "Match system"), ("light", "Light"), ("dark", "Dark")):
            act = QAction(label, self, checkable=True)
            act.setChecked(choice == self.theme_name)
            act.triggered.connect(lambda _c, c=choice: self._set_theme(c))
            self.appearance_group.addAction(act)
            self.appearance_actions[choice] = act

        self.a_run_detection = A("Run detection again", lambda: self._run_detection(), None)
        self.a_split_contacts = A("Split touching animals", lambda on: self._set_split(on), checkable=True)
        self.a_split_contacts.setChecked(True)
        self.a_calibrate = A("Calibrate arena...", lambda: self._start_calibration())
        self.a_draw_zone = A("Draw zone...", lambda: self._start_zone())
        self.a_heatmap = A("Occupancy heatmap...", lambda: self._show_heatmap())
        self.a_transitions = A("Behavior transitions...", lambda: self._show_transitions())
        self.a_batch = A("Batch queue...", lambda: self._show_batch())

        self.a_cheat = A("Keyboard shortcuts", lambda: ShortcutsDialog(self.project, self).exec(), "cheat_sheet")
        self.a_about = A("About SCRUM Viewer", lambda: self._about())
        self.a_about.setMenuRole(QAction.MenuRole.AboutRole)


        hk = {
            "play": self._toggle_play, "step_back": lambda: self._step(-1), "step_fwd": lambda: self._step(1),
            "step_back10": lambda: self._step(-10), "step_fwd10": lambda: self._step(10),
            "speed_up": lambda: self._bump_speed(1), "speed_down": lambda: self._bump_speed(-1),
            "loop": lambda: self._toggle_loop(), "home": lambda: self._seek(0),
            "end": lambda: self._seek(max(0, self.project.n_frames - 1)),
            "prev_bout": lambda: self._goto_bout(-1), "next_bout": lambda: self._goto_bout(1),
            "prev_review": lambda: self._goto_review(-1), "next_review": lambda: self._goto_review(1),
            "prev_bookmark": lambda: self._goto_bookmark(-1), "next_bookmark": lambda: self._goto_bookmark(1),
            "clear_selection": self._escape,
        }
        for sid, handler in hk.items():
            self._act(sid, lambda _checked=False, h=handler: h(), sid)

    def _build_menus(self) -> None:
        mb = self.menuBar()
        f = mb.addMenu("&File")
        f.addAction(self.a_open_video)
        f.addAction(self.a_open_tracks)
        f.addAction(self.a_open_preds)
        self.recent_menu = f.addMenu("Open Recent")
        f.addSeparator()
        f.addAction(self.a_open_session)
        f.addAction(self.a_save_session)
        f.addAction(self.a_save_session_as)
        f.addSeparator()
        ex = f.addMenu("Export")
        for a in (self.a_export_clip, self.a_export_gif, self.a_export_pdf):
            ex.addAction(a)
        ex.addSeparator()
        for a in (self.a_export_labels, self.a_export_bouts, self.a_export_summary, self.a_export_boris):
            ex.addAction(a)
        f.addSeparator()
        f.addAction(self.a_quit)
        self._refresh_recent_menu()

        e = mb.addMenu("&Edit")
        for a in (self.a_undo, self.a_redo):
            e.addAction(a)
        e.addSeparator()
        for a in (self.a_start_manual, self.a_clear_manual):
            e.addAction(a)
        e.addSeparator()
        for a in (self.a_paint, self.a_split, self.a_merge, self.a_delete_bout, self.a_bookmark):
            e.addAction(a)

        v = mb.addMenu("&View")
        self.toggle_actions: dict[str, QAction] = {}
        ov = v.addMenu("Overlays")
        for attr, text in TOGGLES:
            a = QAction(text, self, checkable=True)
            a.setChecked(getattr(self.renderer.options, attr))
            a.toggled.connect(lambda on, at=attr: self._toggle_overlay(at, on))
            ov.addAction(a)
            self.toggle_actions[attr] = a
        mag = v.addMenu("Magnifier zoom")
        grp = QActionGroup(self)
        for z in (2.0, 3.0, 4.0, 6.0):
            a = QAction(f"{z:g}x", self, checkable=True)
            a.setChecked(abs(self.renderer.options.magnifier_zoom - z) < 1e-6)
            a.triggered.connect(lambda _c, zz=z: self._set_option("magnifier_zoom", zz))
            grp.addAction(a)
            mag.addAction(a)
        trail = v.addMenu("Trail length")
        grp2 = QActionGroup(self)
        for sec in (1.0, 3.0, 5.0, 10.0):
            a = QAction(f"{sec:g} s", self, checkable=True)
            a.setChecked(abs(self.renderer.options.trail_seconds - sec) < 1e-6)
            a.triggered.connect(lambda _c, s=sec: self._set_option("trail_seconds", s))
            grp2.addAction(a)
            trail.addAction(a)
        v.addSeparator()
        for a in (self.a_zoom_in, self.a_zoom_out, self.a_zoom_fit):
            v.addAction(a)
        v.addSeparator()
        am = v.addMenu("Appearance")
        for act in self.appearance_actions.values():
            am.addAction(act)
        dm = v.addMenu("Panels")
        for dock in self.docks.values():
            dm.addAction(dock.toggleViewAction())

        an = mb.addMenu("&Analysis")
        for a in (self.a_run_detection, self.a_split_contacts):
            an.addAction(a)
        an.addSeparator()
        for a in (self.a_calibrate, self.a_draw_zone, self.a_heatmap, self.a_transitions):
            an.addAction(a)
        an.addSeparator()
        an.addAction(self.a_batch)

        h = mb.addMenu("&Help")
        h.addAction(self.a_cheat)
        h.addAction(self.a_about)

    def _wire(self) -> None:
        p = self.project
        hints = QApplication.instance().styleHints()
        if hasattr(hints, "colorSchemeChanged"):
            hints.colorSchemeChanged.connect(lambda *_: self._on_system_scheme_changed())
        self.play_button.clicked.connect(self._toggle_play)
        self.step_back_button.clicked.connect(lambda: self._step(-1))
        self.step_fwd_button.clicked.connect(lambda: self._step(1))
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        self.loop_check.toggled.connect(lambda on: self._toggle_loop(on))
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        self.timeline.seek_requested.connect(self._seek)
        self.traces.seek_requested.connect(self._seek)
        self.timeline.view_changed.connect(self.traces.set_view)
        self.timeline.bout_selected.connect(self.annotate_panel.show_bout)
        self.bookmarks_panel.seek_requested.connect(self._seek)
        self.review_panel.items_changed.connect(self.timeline.set_review_items)
        self.review_panel.jump_requested.connect(self._jump_review)
        self.annotate_panel.behavior_chosen.connect(self._set_paint_behavior)
        self.annotate_panel.paint_toggled.connect(self._set_paint)
        self.annotate_panel.start_btn.clicked.connect(p.start_manual_from_predictions)
        self.annotate_panel.clear_btn.clicked.connect(self._clear_manual)
        self.annotate_panel.undo_btn.clicked.connect(self._undo)
        self.annotate_panel.redo_btn.clicked.connect(self._redo)
        self.zones_panel.calibrate_requested.connect(self._start_calibration)
        self.zones_panel.draw_zone_requested.connect(self._start_zone)
        self.zones_panel.clear_calibration_requested.connect(lambda: p.set_calibration(None))
        self.view.point_clicked.connect(self._on_view_point)
        self.view.point_double_clicked.connect(self._on_view_double_click)

        p.labels_changed.connect(self._on_labels_changed)
        p.bookmarks_changed.connect(self._mark_dirty)
        p.zones_changed.connect(self._on_geometry_changed)
        p.calibration_changed.connect(self._on_geometry_changed)
        p.data_changed.connect(self._rebuild_behavior_keys)


    def _confirm_discard(self) -> bool:
        if not self._dirty:
            return True
        box = QMessageBox(self)
        box.setWindowTitle("Unsaved work")
        box.setText("You have edits, bookmarks or zones that are not saved to a session file.")
        save = box.addButton("Save session", QMessageBox.ButtonRole.AcceptRole)
        discard = box.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is save:
            return self.save_session()
        return box.clickedButton() is discard

    def _open_video_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv *.m4v)")
        if path:
            self.load_video(path)

    def _open_tracks_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Tracks / Pose", "", "Tracks and pose (*.csv *.h5 *.hdf5 *.slp)")
        if path:
            self.load_tracks(path)

    def _open_predictions_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Predictions CSV", "", "CSV Files (*.csv)")
        if path:
            self.load_predictions(path)

    def _open_session_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Session", "", f"SCRUM session (*{SESSION_EXT})")
        if path:
            self.load_session(path)

    def load_video(self, path: str, auto_detect: bool = True, confirm: bool = True) -> bool:
        if confirm and not self._confirm_discard():
            return False
        self._cancel_detection()
        self._stop_recording(commit=False)
        try:
            new_video = VideoSource(path)
        except IOError as e:
            self.banner.show_message(f"Could not open {Path(path).name}: {e}", "error")
            self.view.set_message(f"Could not open the video.\n{e}")
            return False
        if self.video is not None:
            self.video.release()
        self.video = new_video
        self.video.frame_ready.connect(self._on_frame)
        self.video.playing_changed.connect(lambda on: self.play_button.setText("Pause" if on else "Play"))
        self.video.finished.connect(self._on_playback_finished)
        self.video.set_speed(self.speed_combo.currentData())
        self.loop_check.blockSignals(True)
        self.loop_check.setChecked(False)
        self.loop_check.blockSignals(False)
        self.session_path = None
        self.project.set_video(path, self.video.fps, self.video.frame_count, self.video.width, self.video.height)
        self._dirty = False
        self.setWindowTitle(f"SCRUM Viewer - {Path(path).name}")
        self.view.set_message("Loading...")
        self._add_recent(path)
        self._seek(0)
        self._update_enabled()

        if self.mode == "scrum":
            if not self._load_cached_result(path) and auto_detect:
                self._run_detection()
        else:
            self._announce_calms21_needs_tracks()
        return True

    def _load_cached_result(self, path: str) -> bool:
        cache = find_cache(path)
        if cache is None:
            return False
        try:
            result = DetectionResult.load(cache)
        except Exception:
            return False
        if not result.matches_video(path) or result.mode != "scrum" or result.n_frames not in (0, self.project.n_frames):
            return False
        self._apply_result(result)
        self.banner.show_message(
            f"Loaded saved results for this video ({cache.name}). Use Analysis > Run detection again to recompute.", "info", timeout_ms=7000
        )
        return True

    def load_tracks(self, path: str) -> None:
        try:
            tracks = load_tracks(path)
        except (ValueError, OSError, ImportError) as e:
            self.banner.show_message(f"Could not load {Path(path).name}: {e}", "error")
            return
        self.project.set_tracks(tracks)
        n_animals = tracks["track_id"].nunique()
        self.banner.show_message(f"Loaded {len(tracks)} track points for {n_animals} animals from {Path(path).name}.", "info", timeout_ms=6000)
        if self.mode == "calms21":
            self._run_keypoint_detection()
        self._refresh_frame()

    def load_predictions(self, path: str) -> None:
        try:
            preds = load_predictions(path)
        except (ValueError, OSError) as e:
            self.banner.show_message(f"Could not load predictions from {Path(path).name}: {e}", "error")
            return
        self.project.set_predictions(preds, f"predictions loaded from {Path(path).name}")
        self.model_label.setText(f"model: {self.project.source}")
        self.banner.show_message(f"Loaded predictions for {len(preds)} frames from {Path(path).name}.", "info", timeout_ms=6000)
        self._refresh_frame()


    def _announce_calms21_needs_tracks(self) -> None:
        self.model_label.setText("model: CalMS21 keypoint model (needs pose tracks)")
        self.view.set_message("CalMS21 mode needs a pose tracks file.\nFile > Open Tracks / Pose..., or drop the file here.")
        self.banner.show_message(
            "CalMS21 mode needs a pose tracks file (a tracks CSV with frame, track_id, keypoint, x, y, a SLEAP .h5 or a DeepLabCut CSV). "
            "It classifies from pose keypoints, not from raw video.",
            "warn", "Open tracks...", self._open_tracks_dialog,
        )

    def _on_mode_changed(self, _i: int) -> None:
        self.mode = self.mode_combo.currentData()
        self._cancel_detection()
        if self.video is None:
            self.model_label.setText("model: SCRUM (raw video), open a video" if self.mode == "scrum" else "model: CalMS21 (pose tracks), open a video")
            return
        self.project.clear_detection()
        if self.mode == "scrum":
            if not self._load_cached_result(self.project.video_path):
                self._run_detection()
        else:
            self._announce_calms21_needs_tracks()
        self._refresh_frame()

    def _set_split(self, on: bool) -> None:
        self.use_split = on
        if self.video is not None and self.mode == "scrum":
            self._run_detection()

    def _cancel_detection(self) -> None:
        self._detect_token += 1
        if self._detect_worker is not None:
            self._detect_worker.cancel()
            self._detect_worker = None

    def _run_detection(self) -> None:
        if self.video is None:
            return
        if self.mode == "calms21":
            if self.project.tracks.empty:
                self._announce_calms21_needs_tracks()
            else:
                self._run_keypoint_detection()
            return
        self._cancel_detection()
        token = self._detect_token
        path = self.project.video_path
        self.model_label.setText("model: running SCRUM detection...")
        self.banner.show_message("Tracking the animals and classifying behavior... 0%", "info", "Cancel", self._cancel_detection_clicked)
        use_split = self.use_split

        from scrum.app.workers import start_worker

        thread, worker = start_worker(
            lambda report: run_scrum_detection_full(path, progress=report, use_split=use_split),
            lambda result, t=token: self._on_detection_finished(t, result),
            lambda msg, t=token: self._on_detection_failed(t, msg),
            lambda d, total, t=token: self._on_detection_progress(t, d, total),
        )
        self._tasks.append((thread, worker))
        self._detect_worker = worker

    def _cancel_detection_clicked(self) -> None:
        self._cancel_detection()
        self.model_label.setText("model: detection cancelled")
        self.banner.show_message("Detection cancelled. The video still plays. Analysis > Run detection again to retry.", "warn", timeout_ms=8000)

    def _on_detection_progress(self, token: int, done: int, total: int) -> None:
        if token != self._detect_token or total <= 0:
            return
        self.banner.show_message(f"Tracking the animals and classifying behavior... {100 * done // total}%", "info", "Cancel", self._cancel_detection_clicked)

    def _on_detection_finished(self, token: int, result: DetectionResult) -> None:
        if token != self._detect_token:
            return
        self._detect_worker = None
        self._apply_result(result)
        for target in (cache_path_for(result.video_path), results_dir_for(result.video_path) / (Path(result.video_path).stem + ".scrum_cache.npz")):
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                result.save(target)
                break
            except OSError:
                continue
        merged_pct = 100.0 * float(result.merged.mean()) if result.merged is not None and len(result.merged) else 0.0
        if result.source.startswith("heuristic"):
            self.banner.show_message(
                f"No trained SCRUM model was used ({result.source}). Labels come from a simple contact heuristic and have no confidence scores. "
                "Put calms21_sbm_centroid_hgb.joblib in the models folder for model predictions.",
                "warn",
            )
        else:
            self.banner.show_message(
                f"Detection finished: {len(self.project.bouts())} bouts. Animals were fused into one blob on {merged_pct:.0f}% of frames.", "info", timeout_ms=8000
            )

    def _on_detection_failed(self, token: int, message: str) -> None:
        if token != self._detect_token:
            return
        self._detect_worker = None
        self.model_label.setText("model: detection failed")
        self.banner.show_message(
            f"The tracker could not process this video: {message}. You can still load tracks and predictions CSVs from the File menu.", "error"
        )

    def _apply_result(self, result: DetectionResult) -> None:
        self.project.set_result(result)
        self.model_label.setText(f"model: {result.source}")
        self._refresh_frame()

    def _run_keypoint_detection(self) -> None:
        if self.project.tracks.empty:
            return
        try:
            preds = run_calms21_keypoint_detection(self.project.tracks, fps=self.project.fps)
        except FileNotFoundError as e:
            self.banner.show_message(str(e), "error")
            return
        except ValueError as e:
            self.banner.show_message(f"Could not build features from these tracks: {e}", "error")
            return
        self.project.set_predictions(preds, "CalMS21 model (full keypoints)")
        self.model_label.setText("model: CalMS21 model (full keypoints)")
        self.banner.show_message(f"Classified {len(preds)} frames from the pose tracks.", "info", timeout_ms=6000)
        self._refresh_frame()


    def _on_frame(self, idx: int, frame: np.ndarray) -> None:
        annotated = self.renderer.render(frame, idx)
        self.view.set_image(bgr_to_qimage(annotated))
        p = self.project
        label = p.label_at(idx)
        self.frame_label.setText(
            f"frame {idx} / {max(0, p.n_frames - 1)}   {fmt_time(idx, p.fps)}" + (f"   {label}" if label and p.has_predictions or (label and p.n_manual_frames()) else "")
        )
        self.timeline.set_current_frame(idx)
        self.traces.set_current_frame(idx)
        if self._rec:
            self._update_recording(idx)

    def _refresh_frame(self) -> None:
        if self.video is None:
            return
        frame = self.video.read_frame(self.video.current_frame)
        if frame is not None:
            self._on_frame(self.video.current_frame, frame)

    def _toggle_play(self) -> None:
        if self.video is None:
            return
        if self.video.is_playing():
            self.video.pause()
        else:
            self.video.play()

    def _step(self, delta: int) -> None:
        if self.video is None:
            return
        self.video.pause()
        self.video.step(delta)

    def _seek(self, frame: int) -> None:
        if self.video is not None:
            self.video.seek(int(frame))

    def _on_speed_changed(self, _i: int) -> None:
        if self.video is not None:
            self.video.set_speed(self.speed_combo.currentData())

    def _bump_speed(self, d: int) -> None:
        i = max(0, min(self.speed_combo.count() - 1, self.speed_combo.currentIndex() + d))
        self.speed_combo.setCurrentIndex(i)

    def _on_playback_finished(self) -> None:
        if self._rec:
            self._stop_recording(commit=True)


    def _loop_range(self) -> tuple[int, int] | None:
        if self.timeline.selection:
            return self.timeline.selection
        b = self.timeline.selected_bout or (self.project.bout_at(self.video.current_frame) if self.video else None)
        return (b["start"], b["end"]) if b else None

    def _toggle_loop(self, on: bool | None = None) -> None:
        if self.video is None:
            return
        on = (not self.video.loop_enabled) if on is None else on
        if on:
            rng = self._loop_range()
            if rng is None:
                self._status("Select a range (Shift+drag on the timeline) or a bout to loop it.")
                on = False
            else:
                self.video.set_loop(True, rng)
                self.timeline.set_loop_range(rng)
                if not rng[0] <= self.video.current_frame <= rng[1]:
                    self._seek(rng[0])
                if not self.video.is_playing():
                    self.video.play()
        if not on:
            self.video.set_loop(False)
            self.timeline.set_loop_range(None)
        self.loop_check.blockSignals(True)
        self.loop_check.setChecked(on)
        self.loop_check.blockSignals(False)


    def _goto_bout(self, direction: int) -> None:
        if self.video is None:
            return
        f = self.video.current_frame
        bouts = self.project.bouts()
        nxt = next((b for b in bouts if b["start"] > f), None) if direction > 0 else next((b for b in reversed(bouts) if b["start"] < f), None)
        if nxt is None:
            self._status("No more bouts in that direction.")
            return
        self.timeline.set_selected_bout(nxt)
        self._seek(nxt["start"])

    def _goto_bookmark(self, direction: int) -> None:
        if self.video is None:
            return
        f = self.video.current_frame
        marks = self.project.bookmarks
        nxt = next((b for b in marks if b["frame"] > f), None) if direction > 0 else next((b for b in reversed(marks) if b["frame"] < f), None)
        if nxt is None:
            self._status("No more bookmarks in that direction.")
            return
        self._seek(nxt["frame"])

    def _goto_review(self, direction: int) -> None:
        if self.video is not None:
            self.review_panel.step(direction, from_frame=self.video.current_frame)

    def _jump_review(self, start: int, end: int) -> None:
        self._seek(start)
        self.timeline.set_selection((start, end))
        self.timeline.ensure_visible(start)


    def _set_paint(self, on: bool) -> None:
        self.timeline.paint_mode = on
        for w, sig in ((self.a_paint, None), (self.annotate_panel.paint, None)):
            w.blockSignals(True)
            w.setChecked(on)
            w.blockSignals(False)
        self._status("Paint tool on: drag on the timeline to add a bout of the active behavior." if on else "Paint tool off.")

    def _set_paint_behavior(self, name: str) -> None:
        self.timeline.paint_behavior = name

    def _undo(self) -> None:
        text = self.project.undo()
        self._status(f"Undid: {text}" if text else "Nothing to undo.")

    def _redo(self) -> None:
        text = self.project.redo()
        self._status(f"Redid: {text}" if text else "Nothing to redo.")

    def _clear_manual(self) -> None:
        if self.project.n_manual_frames() == 0:
            return
        if QMessageBox.question(self, "Discard edits", "Discard every manual edit and go back to the model's labels? You can undo this.") == QMessageBox.StandardButton.Yes:
            self.project.clear_manual()

    def _selected_bout(self) -> dict | None:
        b = self.timeline.selected_bout
        if b is None and self.video is not None:
            b = self.project.bout_at(self.video.current_frame)
        return b

    def _delete_selected_bout(self) -> None:
        b = self._selected_bout()
        if b and self.project.delete_bout(b):
            self.timeline.set_selected_bout(None)

    def _split_at_playhead(self) -> None:
        if self.video and not self.project.split_bout(self.video.current_frame):
            self._status("Nothing to split here: park the playhead inside a bout, away from its ends.")

    def _merge_selected(self) -> None:
        b = self._selected_bout()
        if b and self.project.merge_with_next(b):
            self.timeline.set_selected_bout(self.project.bout_at(b["start"]))
        else:
            self._status("There is no later bout to merge with.")

    def _bookmark_here(self) -> None:
        if self.video is not None:
            self.project.add_bookmark(self.video.current_frame, "")
            self._status(f"Bookmarked frame {self.video.current_frame}. Add a note in the Bookmarks panel.")

    def _escape(self) -> None:
        if self._pick is not None:
            self._end_pick("Cancelled.")
        elif self._rec:
            self._stop_recording(commit=True)
        elif self.timeline.selection:
            self.timeline.set_selection(None)


    def _rebuild_behavior_keys(self) -> None:
        for a in self._behavior_actions:
            self.removeAction(a)
            a.deleteLater()
        self._behavior_actions = []
        for beh, key in shortcuts.behavior_key_map(self.project.behaviors()).items():
            a = QAction(f"Label {beh}", self)
            a.setShortcut(QKeySequence(key))
            a.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            a.triggered.connect(lambda _c=False, b=beh: self._on_behavior_key(b))
            self.addAction(a)
            self._behavior_actions.append(a)

    def _on_behavior_key(self, behavior: str) -> None:
        if self.video is None:
            return
        sel = self.timeline.selection
        if sel:
            self.project.set_range(sel[0], sel[1], behavior)
            self.timeline.set_selection(None)
            self._status(f"Labeled frames {sel[0]} to {sel[1]} as {behavior}.")
            return
        frame = self.video.current_frame
        if self._rec:
            rec_b, start = self._rec
            if rec_b == behavior:
                self._stop_recording(commit=True)
                return
            self._stop_recording(commit=True, end=max(start, frame - 1))
        self._rec = (behavior, frame)
        self._update_recording(frame)
        self._status(f"Recording {behavior} from frame {frame}. Press the key again (or Esc) to end it.")

    def _update_recording(self, idx: int) -> None:
        if not self._rec:
            return
        beh, start = self._rec
        end = max(start, idx)
        self.timeline.set_recording((start, end, beh))
        self.annotate_panel.show_recording(f"REC {beh}: frames {start} to {end}")

    def _stop_recording(self, commit: bool = True, end: int | None = None) -> None:
        if not self._rec:
            return
        beh, start = self._rec
        self._rec = None
        self.timeline.set_recording(None)
        self.annotate_panel.show_recording("")
        if commit and self.video is not None:
            stop = max(start, self.video.current_frame if end is None else end)
            if self.project.set_range(start, stop, beh):
                self.timeline.set_selected_bout(self.project.bout_at(start))
                self._status(f"Recorded {beh}: frames {start} to {stop}.")


    def _start_pick(self, kind: str, hint: str) -> None:
        if self.video is None:
            self._status("Open a video first.")
            return
        self._playing_before_pick = self.video.is_playing()
        self.video.pause()
        self._pick = {"kind": kind, "points": []}
        self.view.set_pick_mode(True)
        self.view.set_hint(hint)
        self.view.set_pending([])

    def _end_pick(self, message: str = "") -> None:
        self._pick = None
        self.view.set_pick_mode(False)
        if message:
            self._status(message)

    def _start_calibration(self) -> None:
        self._start_pick("calibrate", f"Calibrate: click the arena's {CORNER_NAMES[0]} corner (1 of 4). Esc cancels.")

    def _start_zone(self) -> None:
        self._start_pick("zone", "Draw zone: click each corner, double-click to finish. Esc cancels.")

    def _on_view_point(self, x: float, y: float) -> None:
        if not self._pick:
            return
        pts = self._pick["points"]
        pts.append((x, y))
        self.view.set_pending(pts, closed=self._pick["kind"] == "calibrate" and len(pts) == 4)
        if self._pick["kind"] == "calibrate":
            if len(pts) < 4:
                self.view.set_hint(f"Calibrate: click the arena's {CORNER_NAMES[len(pts)]} corner ({len(pts) + 1} of 4). Esc cancels.")
            else:
                self._finish_calibration(list(pts))
        else:
            self.view.set_hint(f"Draw zone: {len(pts)} points. Double-click to finish. Esc cancels.")

    def _on_view_double_click(self, _x: float, _y: float) -> None:
        if self._pick and self._pick["kind"] == "zone":
            pts = list(self._pick["points"])
            if len(pts) < 3:
                self._end_pick("A zone needs at least three corners.")
                return
            name, ok = QInputDialog.getText(self, "Zone name", "Name for this zone:", text=f"zone {len(self.project.zones) + 1}")
            self._end_pick()
            if ok and name.strip():
                self.project.add_zone(name.strip(), pts)
                self._status(f"Added zone '{name.strip()}'. Times appear in the Zones panel.")

    def _finish_calibration(self, corners: list[tuple[float, float]]) -> None:
        width, ok = QInputDialog.getDouble(self, "Arena width", "Real width of the arena (top edge, in cm):", 50.0, 0.1, 10000.0, 1)
        if not ok:
            self._end_pick("Calibration cancelled.")
            return
        height, ok2 = QInputDialog.getDouble(self, "Arena height", "Real height (left edge, in cm). 0 to use the width only:", 0.0, 0.0, 10000.0, 1)
        self._end_pick()
        cal = Calibration(corners, float(width), float(height) if ok2 and height > 0 else None)
        self.project.set_calibration(cal)
        self._status(f"Calibrated: {cal.px_per_cm:.2f} px per cm. Distances and speeds are now in cm.")

    def _background_frame(self) -> np.ndarray | None:
        if self.video is None:
            return None
        return self.video.read_frame(self.project.n_frames // 2) or None

    def _show_heatmap(self) -> None:
        if not self.project.positions:
            self.banner.show_message("The heatmap needs tracked animals. Run detection or open a tracks file first.", "warn", timeout_ms=6000)
            return
        bg = self._background_frame()
        HeatmapDialog(self.project, bg, self).exec()
        self._refresh_frame()

    def _show_transitions(self) -> None:
        if len(self.project.bouts()) < 2:
            self.banner.show_message("Transitions need at least two bouts.", "warn", timeout_ms=6000)
            return
        TransitionDialog(self.project, self).exec()

    def _show_batch(self, folder: str | None = None) -> None:
        if self._batch is None:
            self._batch = BatchWindow()
            self._batch.open_requested.connect(lambda path: self.load_video(path))
        self._batch.show()
        self._batch.raise_()
        if folder:
            self._batch.add_paths([folder])


    def _export_range(self) -> tuple[int, int]:
        rng = self.timeline.selection or (
            (self.timeline.selected_bout["start"], self.timeline.selected_bout["end"]) if self.timeline.selected_bout else None
        )
        return rng if rng else (0, max(0, self.project.n_frames - 1))

    def _default_name(self, suffix: str) -> str:
        stem = Path(self.project.video_path).stem if self.project.video_path else "scrum"
        return str(Path(self.project.video_path).with_name(f"{stem}{suffix}")) if self.project.video_path else f"{stem}{suffix}"

    def _run_task(self, title: str, fn, on_done) -> None:
        from scrum.app.workers import start_worker

        dlg = QProgressDialog(title, "Cancel", 0, 100, self)
        dlg.setWindowTitle("SCRUM")
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setValue(0)
        dlg.show()

        def done(result):
            dlg.close()
            on_done(result)

        def failed(msg):
            dlg.close()
            self.banner.show_message(f"{title} failed: {msg}", "error")

        def progress(d, total):
            if total > 0:
                dlg.setValue(int(100 * d / total))

        thread, worker = start_worker(fn, done, failed, progress)
        dlg.canceled.connect(worker.cancel)
        self._tasks.append((thread, worker))

    def _export_clip(self, gif: bool) -> None:
        if not self.project.has_video:
            return
        s, e = self._export_range()
        what = "selection" if (self.timeline.selection or self.timeline.selected_bout) else "whole video"
        if gif:
            path, _ = QFileDialog.getSaveFileName(self, f"Export GIF of {what}", self._default_name(f"_{s}-{e}.gif"), "GIF (*.gif)")
        else:
            path, _ = QFileDialog.getSaveFileName(self, f"Export clip of {what}", self._default_name(f"_{s}-{e}.mp4"), "MP4 (*.mp4);;AVI (*.avi)")
        if not path:
            return
        renderer = FrameRenderer(self.project, OverlayOptions.from_dict(self.renderer.options.to_dict()))
        speed = self.speed_combo.currentData()
        proj = self.project
        job = (lambda report: export.export_gif(proj, renderer, path, s, e, speed=speed, progress=report)) if gif else (
            lambda report: export.export_clip(proj, renderer, path, s, e, speed=speed, progress=report)
        )
        self._run_task("Exporting GIF..." if gif else "Exporting clip...", job, lambda n: self.banner.show_message(f"Wrote {n} frames to {path}", "info", timeout_ms=9000))

    def _export_table(self, kind: str) -> None:
        if not self.project.has_video:
            return
        names = {"labels": "_labels.csv", "bouts": "_bouts.csv", "summary": "_summary.csv", "boris": "_boris_events.csv"}
        path, _ = QFileDialog.getSaveFileName(self, "Export", self._default_name(names[kind]), "CSV (*.csv)")
        if not path:
            return
        try:
            if kind == "labels":
                export.export_labels_csv(self.project, path)
            elif kind == "bouts":
                export.export_bouts_csv(self.project, path)
            elif kind == "summary":
                export.export_summary_csv(self.project, path)
            else:
                export.export_boris_csv(self.project, path)
        except OSError as e:
            self.banner.show_message(f"Could not write {path}: {e}", "error")
            return
        self.banner.show_message(f"Wrote {path}", "info", timeout_ms=7000)

    def _export_pdf(self) -> None:
        if not self.project.has_video:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export PDF report", self._default_name("_report.pdf"), "PDF (*.pdf)")
        if not path:
            return
        renderer = FrameRenderer(self.project, OverlayOptions.from_dict(self.renderer.options.to_dict()))
        proj = self.project
        self._run_task("Building the PDF report...", lambda report: export.export_pdf_report(proj, path, renderer), lambda _r: self.banner.show_message(f"Wrote {path}", "info", timeout_ms=9000))


    def save_session(self, ask: bool = False) -> bool:
        if not self.project.has_video:
            return False
        path = self.session_path
        if ask or not path:
            suggested = path or self._default_name(SESSION_EXT)
            path, _ = QFileDialog.getSaveFileName(self, "Save session", suggested, f"SCRUM session (*{SESSION_EXT})")
            if not path:
                return False
            if not path.endswith(SESSION_EXT):
                path += SESSION_EXT
        extra = {
            "overlays": self.renderer.options.to_dict(),
            "playhead": self.video.current_frame if self.video else 0,
            "speed": self.speed_combo.currentData(),
        }
        try:
            self.project.save_session(path, extra)
        except OSError as e:
            self.banner.show_message(f"Could not save the session: {e}", "error")
            return False
        self.session_path = path
        self._dirty = False
        self._add_recent(path)
        self.banner.show_message(f"Session saved to {path}", "info", timeout_ms=5000)
        return True

    def load_session(self, path: str) -> bool:
        if not self._confirm_discard():
            return False
        try:
            d = Project.read_session(path)
        except (ValueError, OSError) as e:
            self.banner.show_message(f"Could not open the session: {e}", "error")
            return False
        video = d.get("video_path")
        if not video or not Path(video).exists():
            video, _ = QFileDialog.getOpenFileName(self, f"Locate the video for {Path(path).name}", "", "Video Files (*.mp4 *.avi *.mov *.mkv *.m4v)")
            if not video:
                return False
        self.mode_combo.blockSignals(True)
        self.mode = d.get("mode", "scrum")
        self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(self.mode)))
        self.mode_combo.blockSignals(False)
        if not self.load_video(video, auto_detect=False, confirm=False):
            return False
        cache = Path(path).parent / (d.get("cache") or "")
        if d.get("cache") and cache.exists():
            try:
                self._apply_result(DetectionResult.load(cache))
            except Exception as e:
                self.banner.show_message(f"The session's saved results could not be read ({e}). Run detection again.", "warn")
        else:
            self.banner.show_message("The session's saved results file is missing. Run detection again (Analysis menu).", "warn")
        self.project.apply_session_dict(d)
        if d.get("overlays"):
            self.renderer.options = OverlayOptions.from_dict(d["overlays"])
            for attr, act in self.toggle_actions.items():
                act.blockSignals(True)
                act.setChecked(getattr(self.renderer.options, attr))
                act.blockSignals(False)
        self.session_path = path
        self._dirty = False
        self._add_recent(path)
        if d.get("playhead"):
            self._seek(int(d["playhead"]))
        self._refresh_frame()
        self.banner.show_message(f"Session {Path(path).name} restored.", "info", timeout_ms=5000)
        return True


    def _recent(self) -> list[str]:
        v = self.settings.value("recent", [])
        return [v] if isinstance(v, str) else list(v or [])

    def _add_recent(self, path: str) -> None:
        items = [p for p in self._recent() if p != path]
        self.settings.setValue("recent", [path] + items[: MAX_RECENT - 1])
        self._refresh_recent_menu()

    def _refresh_recent_menu(self) -> None:
        self.recent_menu.clear()
        items = [p for p in self._recent() if Path(p).exists()]
        if not items:
            a = self.recent_menu.addAction("(none)")
            a.setEnabled(False)
            return
        for p in items:
            self.recent_menu.addAction(Path(p).name, lambda path=p: self._open_recent(path)).setToolTip(p)
        self.recent_menu.addSeparator()
        self.recent_menu.addAction("Clear list", lambda: (self.settings.setValue("recent", []), self._refresh_recent_menu()))

    def _open_recent(self, path: str) -> None:
        if path.endswith(SESSION_EXT):
            self.load_session(path)
        else:
            self.load_video(path)


    def _toggle_overlay(self, attr: str, on: bool) -> None:
        self._set_option(attr, on)

    def _set_option(self, attr: str, value) -> None:
        setattr(self.renderer.options, attr, value)
        self.settings.setValue("overlays", json.dumps(self.renderer.options.to_dict()))
        self._refresh_frame()

    def _set_theme(self, name: str, save: bool = True) -> None:
        if name not in CHOICES:
            name = "system"
        self.theme_name = name
        self._restyle(apply_theme(QApplication.instance(), name))
        if save:
            self.settings.setValue("theme", name)
        if hasattr(self, "appearance_actions"):
            self.appearance_actions[name].setChecked(True)

    def _restyle(self, theme) -> None:
        self.theme = theme
        for w in (self.timeline, self.traces, self.view, self.banner):
            w.set_theme(theme)
        self._refresh_frame()

    def _on_system_scheme_changed(self) -> None:
        if self.theme_name == "system":
            QTimer.singleShot(0, lambda: self._restyle(apply_theme(QApplication.instance(), "system")))

    def _on_labels_changed(self) -> None:
        if self.project.n_manual_frames() or self.project.can_undo():
            self._mark_dirty()
        self._update_enabled()
        if self.video is not None and not self.video.is_playing():
            self._refresh_frame()

    def _on_geometry_changed(self) -> None:
        self._mark_dirty()
        if self.video is not None and not self.video.is_playing():
            self._refresh_frame()

    def _mark_dirty(self) -> None:
        if self.project.has_video:
            self._dirty = True

    def _update_enabled(self) -> None:
        has = self.project.has_video
        for a in (self.a_export_clip, self.a_export_gif, self.a_export_labels, self.a_export_bouts, self.a_export_summary,
                  self.a_export_boris, self.a_export_pdf, self.a_save_session, self.a_save_session_as, self.a_calibrate,
                  self.a_draw_zone, self.a_heatmap, self.a_transitions, self.a_run_detection):
            a.setEnabled(has)
        self.a_undo.setEnabled(self.project.can_undo())
        self.a_redo.setEnabled(self.project.can_redo())
        self.a_clear_manual.setEnabled(self.project.n_manual_frames() > 0)

    def _status(self, text: str, ms: int = 5000) -> None:
        self.statusBar().showMessage(text, ms)

    def _about(self) -> None:
        from scrum import __version__
        from scrum.paths import user_models_dir

        QMessageBox.about(
            self, "About SCRUM Viewer",
            f"SCRUM Viewer {__version__}\n\nTracks two animals in raw video with a blob tracker, classifies their "
            "social behavior, and lets you review and correct the result.\n\nFile > Open Video to begin, or drop a "
            "video, a folder of videos, or a saved session on this window.\n\n"
            f"Newer models can be placed in:\n{user_models_dir()}",
        )


    def dragEnterEvent(self, ev) -> None:
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev) -> None:
        self.open_paths([u.toLocalFile() for u in ev.mimeData().urls() if u.isLocalFile()])

    def open_paths(self, paths: list[str]) -> None:
        videos = [p for p in paths if Path(p).suffix.lower() in VIDEO_EXTENSIONS]
        for p in paths:
            path = Path(p)
            if path.is_dir():
                self._show_batch(str(path))
                self.banner.show_message(f"Added the videos in {path.name} to the batch queue.", "info", timeout_ms=6000)
                return
        if len(videos) > 1:
            self._show_batch()
            self._batch.add_paths(videos)
            return
        for p in paths:
            suffix = Path(p).suffix.lower()
            if suffix == SESSION_EXT:
                self.load_session(p)
            elif suffix in VIDEO_EXTENSIONS:
                self.load_video(p)
            elif suffix in (".h5", ".hdf5", ".slp"):
                self.load_tracks(p)
            elif suffix == ".csv":
                try:
                    kind = csv_kind(p)
                except Exception as e:
                    self.banner.show_message(f"Could not read {Path(p).name}: {e}", "error")
                    continue
                (self.load_predictions if kind == "predictions" else self.load_tracks)(p)
            else:
                self.banner.show_message(f"Don't know how to open {Path(p).name}. Drop a video, a tracks or predictions CSV, a SLEAP .h5, or a .scrum session.", "warn")


    def closeEvent(self, ev) -> None:
        if not self._confirm_discard():
            ev.ignore()
            return
        self._cancel_detection()
        if self._batch is not None:
            self._batch.close()
        for _thread, worker in self._tasks:
            worker.cancel()
        for thread, _worker in self._tasks:
            try:
                thread.quit()
                thread.wait(2000)
            except RuntimeError:
                pass
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        if self.video is not None:
            self.video.release()
        ev.accept()
