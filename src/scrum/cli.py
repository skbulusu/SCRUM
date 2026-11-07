from __future__ import annotations

import argparse
import importlib
import platform
import sys
from pathlib import Path

from scrum import __version__

SUBCOMMANDS = {"gui", "detect", "doctor", "models"}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scrum", description="SCRUM: rodent tracking and behavior classification")
    p.add_argument("--version", action="version", version=f"SCRUM {__version__}")
    sub = p.add_subparsers(dest="cmd", metavar="command")

    g = sub.add_parser("gui", help="open the viewer (the default)")
    g.add_argument("paths", nargs="*", help="files to open: videos, a folder, tracks, predictions or a session")
    g.add_argument("--video", help="video to open")
    g.add_argument("--tracks", help="tracks CSV, SLEAP .h5 or DeepLabCut CSV to load")
    g.add_argument("--predictions", help="predictions CSV to load")
    g.add_argument("--session", help="a saved .scrum session to restore")
    g.add_argument("--folder", help="folder of videos to open in the batch queue")

    d = sub.add_parser("detect", help="track and classify videos without a window")
    d.add_argument("paths", nargs="+", help="video files or folders of videos")
    d.add_argument("--out-dir", help="write results here instead of a scrum_results folder beside each video")
    d.add_argument("--no-cache", action="store_true", help="recompute even if saved results exist")

    sub.add_parser("doctor", help="check dependencies, models and the display setup")
    m = sub.add_parser("models", help="show where SCRUM looks for model files, or add one")
    m.add_argument("--add", metavar="FILE", help="copy a trained .joblib model into your user models folder")
    return p


def cmd_gui(args, unknown) -> int:
    from scrum.app.main import run_gui

    return run_gui(args.video, args.tracks, args.predictions, args.session, args.folder, qt_args=unknown, paths=args.paths)


def cmd_detect(args) -> int:
    try:
        from scrum.app.batch import find_videos, process_video
    except ImportError as e:
        print(f"scrum detect needs the GUI extras ({e}). Install with:\n    pip install \"scrum-behavior[gui]\"", file=sys.stderr)
        return 1

    videos: list[str] = []
    for raw in args.paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            videos += find_videos(path)
        elif path.is_file():
            videos.append(str(path))
        else:
            print(f"Not found: {raw}", file=sys.stderr)
            return 2
    if not videos:
        print("No videos found.", file=sys.stderr)
        return 2

    failures = 0
    for i, video in enumerate(videos, 1):
        label = f"[{i}/{len(videos)}] {Path(video).name}"
        last = {"pct": -1}

        def progress(done, total, label=label, last=last):
            pct = int(100 * done / total) if total else 0
            if pct != last["pct"] and pct % 10 == 0:
                last["pct"] = pct
                print(f"{label}  {pct}%", flush=True)
            return None

        try:
            row = process_video(video, use_cache=not args.no_cache, progress=progress, out_dir=args.out_dir)
        except Exception as e:
            failures += 1
            print(f"{label}  FAILED: {e}", file=sys.stderr)
            continue
        print(f"{label}  {row['status']}: {row.get('bouts', 0)} bouts, {row.get('merged_pct', 0)}% fused, {row.get('source', '')}", flush=True)
    return 1 if failures else 0


def cmd_models(args=None) -> int:
    import shutil

    from scrum import paths

    if args is not None and args.add:
        src = Path(args.add).expanduser()
        if not src.is_file() or src.suffix != ".joblib":
            print(f"Expected a .joblib model file, got: {src}", file=sys.stderr)
            return 2
        dest = paths.user_models_dir()
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest / src.name)
        print(f"Copied {src.name} to {dest}")
        print("Names SCRUM uses: calms21_sbm_centroid_hgb.joblib (raw video) and calms21_sbm_hgb.joblib (pose keypoints).")
        return 0

    print("Models are searched in this order (first match wins):")
    for d in paths.model_dirs():
        found = sorted(f.name for f in d.glob("*.joblib")) if d.is_dir() else []
        print(f"  {d}   {', '.join(found) if found else '(no models)' if d.is_dir() else '(missing)'}")
    print(f"\nPut a newer model in: {paths.user_models_dir()}")
    return 0


def _version_of(module: str) -> str | None:
    try:
        m = importlib.import_module(module)
    except Exception:
        return None
    return str(getattr(m, "__version__", None) or getattr(m, "VERSION", None) or "installed")


def cmd_doctor() -> int:
    from scrum import paths

    print(f"SCRUM {__version__}   Python {platform.python_version()}   {platform.platform()}")
    print(f"frozen app: {paths.is_frozen()}\n")
    required = [("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"), ("scikit-learn", "sklearn"),
                ("joblib", "joblib"), ("h5py", "h5py"), ("opencv", "cv2")]
    gui = [("PyQt6", "PyQt6.QtCore"), ("matplotlib", "matplotlib"), ("reportlab", "reportlab"), ("pillow", "PIL")]
    problems = 0
    print("Core dependencies")
    for name, mod in required:
        v = _version_of(mod)
        print(f"  {'ok     ' if v else 'MISSING'} {name} {v or ''}")
        problems += v is None
    print("Viewer dependencies")
    for name, mod in gui:
        v = _version_of(mod)
        print(f"  {'ok     ' if v else 'MISSING'} {name} {v or ''}")
        problems += v is None
    if _version_of("PyQt6.QtCore"):
        from PyQt6.QtCore import QT_VERSION_STR

        print(f"  Qt {QT_VERSION_STR}")
    if sys.platform.startswith("linux"):
        import ctypes.util

        if not ctypes.util.find_library("xcb-cursor"):
            print("  WARNING libxcb-cursor is missing. Qt 6.5+ needs it to open a window.")
            print("          Ubuntu/Debian: sudo apt install libxcb-cursor0")
            print("          Fedora:        sudo dnf install xcb-util-cursor")
    print("\nModels")
    for fname, what in (("calms21_sbm_centroid_hgb.joblib", "raw-video (centroid) model"),
                        ("calms21_sbm_hgb.joblib", "pose keypoint model")):
        p = paths.find_model(fname)
        print(f"  {'ok     ' if p.is_file() else 'missing'} {what}: {p}")
    print(f"  models are also read from: {paths.user_models_dir()}")
    print("\n" + ("Problems found above." if problems else "Everything needed is installed."))
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or (argv[0] not in SUBCOMMANDS and argv[0] not in ("-h", "--help", "--version")):
        argv = ["gui"] + argv
    parser = _build_parser()
    args, unknown = parser.parse_known_args(argv)
    if args.cmd != "gui" and unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    if args.cmd == "gui":
        return cmd_gui(args, unknown)
    if args.cmd == "detect":
        return cmd_detect(args)
    if args.cmd == "doctor":
        return cmd_doctor()
    if args.cmd == "models":
        return cmd_models(args)
    parser.print_help()
    return 0


def gui_main() -> int:
    return main(["gui"] + sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
