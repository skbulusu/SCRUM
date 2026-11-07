from __future__ import annotations

BEHAVIOR_KEYS = {"sniffing": "S", "biting": "B", "mounting": "M", "none": "N"}
EXTRA_BEHAVIOR_KEYS = ["5", "6", "7", "8", "9"]


SHORTCUTS: list[tuple[str, str, str, str]] = [
    ("play", "Space", "Playback", "Play or pause"),
    ("step_back", "Left", "Playback", "Step back one frame"),
    ("step_fwd", "Right", "Playback", "Step forward one frame"),
    ("step_back10", "Shift+Left", "Playback", "Step back 10 frames"),
    ("step_fwd10", "Shift+Right", "Playback", "Step forward 10 frames"),
    ("speed_up", "Up", "Playback", "Faster (0.25x, 0.5x, 1x, 2x, 4x)"),
    ("speed_down", "Down", "Playback", "Slower"),
    ("loop", "L", "Playback", "Loop the selection (or the current bout)"),
    ("home", "Home", "Playback", "Jump to the start"),
    ("end", "End", "Playback", "Jump to the end"),
    ("prev_bout", "[", "Navigate", "Previous bout"),
    ("next_bout", "]", "Navigate", "Next bout"),
    ("prev_review", "{", "Navigate", "Previous review item (low confidence or merged blob)"),
    ("next_review", "}", "Navigate", "Next review item"),
    ("prev_bookmark", "Ctrl+[", "Navigate", "Previous bookmark"),
    ("next_bookmark", "Ctrl+]", "Navigate", "Next bookmark"),
    ("zoom_in", "Ctrl+=", "Timeline", "Zoom the timeline in"),
    ("zoom_out", "Ctrl+-", "Timeline", "Zoom the timeline out"),
    ("zoom_fit", "Ctrl+0", "Timeline", "Fit the whole video"),
    ("undo", "Ctrl+Z", "Annotate", "Undo"),
    ("redo", "Ctrl+Shift+Z", "Annotate", "Redo"),
    ("delete_bout", "Delete", "Annotate", "Delete the selected bout"),
    ("split_bout", "X", "Annotate", "Split the bout at the playhead"),
    ("merge_bout", "J", "Annotate", "Merge the selected bout with the next one"),
    ("paint", "P", "Annotate", "Toggle the paint tool (drag on the timeline to add a bout)"),
    ("bookmark", "K", "Annotate", "Bookmark this frame"),
    ("clear_selection", "Escape", "Annotate", "Clear the selection or stop recording"),
    ("save_session", "Ctrl+S", "File", "Save session"),
    ("open_video", "Ctrl+O", "File", "Open video"),
    ("cheat_sheet", "?", "Help", "Show this cheat sheet"),
]

MOUSE_HELP = [
    ("Timeline", "Click or drag", "Seek"),
    ("Timeline", "Shift+drag", "Select a range (loop it, label it, export it)"),
    ("Timeline", "Drag a bout edge in the Final row", "Resize the bout"),
    ("Timeline", "Alt+drag", "Paint a bout with the active behavior"),
    ("Timeline", "Double-click a bout", "Select it"),
    ("Timeline", "Right-click", "Split, merge, relabel, delete, bookmark"),
    ("Timeline", "Mouse wheel / Ctrl+wheel", "Scroll / zoom"),
    ("Timeline", "Drag the overview", "Move the visible window"),
]

RECORDING_HELP = (
    "Behavior keys work like BORIS. With a range selected, pressing a behavior key labels the "
    "selection. With nothing selected, the first press starts a bout at the playhead (while the video plays "
    "or as you step), and pressing the same key again, a different behavior key, or Escape ends it."
)


def behavior_key_map(behaviors: list[str]) -> dict[str, str]:
    keys = dict(BEHAVIOR_KEYS)
    extra = iter(EXTRA_BEHAVIOR_KEYS)
    out = {}
    for b in behaviors:
        if b in keys:
            out[b] = keys[b]
        else:
            k = next(extra, None)
            if k:
                out[b] = k
    return out
