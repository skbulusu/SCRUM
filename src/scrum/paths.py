from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "SCRUM"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def user_data_dir() -> Path:
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / APP_NAME
    if sys.platform.startswith("win"):
        return Path(os.environ.get("APPDATA") or home / "AppData" / "Roaming") / APP_NAME
    return Path(os.environ.get("XDG_DATA_HOME") or home / ".local" / "share") / APP_NAME.lower()


def user_models_dir() -> Path:
    return user_data_dir() / "models"


def bundled_models_dir() -> Path:
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "scrum" / "models"
    return Path(__file__).resolve().parent / "models"


def legacy_models_dir() -> Path:
    return Path.home() / ".scrum" / "models"


def model_dirs() -> list[Path]:
    dirs: list[Path] = []
    env = os.environ.get("SCRUM_MODELS_DIR")
    if env:
        dirs.append(Path(env).expanduser())
    dirs += [user_models_dir(), legacy_models_dir(), bundled_models_dir()]
    return dirs


def find_model(filename: str) -> Path:
    for d in model_dirs():
        cand = d / filename
        if cand.is_file():
            return cand
    return bundled_models_dir() / filename


def asset_path(name: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", "")) / "scrum" if is_frozen() else Path(__file__).resolve().parent
    return base / "assets" / name
