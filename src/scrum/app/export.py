from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from scrum.app import analysis
from scrum.app.analysis import BACKGROUND
from scrum.app.data import behavior_rgb
from scrum.app.project import Project


def labels_table(project: Project) -> pd.DataFrame:
    n = project.n_frames
    fps = project.fps or 30.0
    pred = np.where(project.pred == "", BACKGROUND, project.pred)
    df = pd.DataFrame(
        {
            "frame": np.arange(n),
            "time_s": np.arange(n) / fps,
            "behavior": project.final_labels(),
            "model_behavior": pred,
            "confidence": project.conf,
            "edited": project.manual != "",
            "merged_blob": project.merged if len(project.merged) == n else False,
        }
    )
    if len(project.distance) == n:
        df[f"inter_animal_distance_{project.units}"] = project.distance
    for a, v in project.speed.items():
        if len(v) == n:
            df[f"{a}_speed_{project.units}_per_s"] = v
    return df


def bouts_table(project: Project) -> pd.DataFrame:
    fps = project.fps or 30.0
    rows = []
    for b in project.bouts():
        s, e = b["start"], b["end"]
        conf = project.conf[s : e + 1]
        rows.append(
            {
                "behavior": b["behavior"],
                "start_frame": s,
                "end_frame": e,
                "start_s": s / fps,
                "end_s": (e + 1) / fps,
                "duration_s": (e - s + 1) / fps,
                "mean_confidence": float(np.nanmean(conf)) if len(conf) and not np.all(np.isnan(conf)) else np.nan,
                "edited": bool((project.manual[s : e + 1] != "").any()),
            }
        )
    return pd.DataFrame(
        rows, columns=["behavior", "start_frame", "end_frame", "start_s", "end_s", "duration_s", "mean_confidence", "edited"]
    )


def export_labels_csv(project: Project, path: str | Path) -> None:
    labels_table(project).to_csv(path, index=False)


def export_bouts_csv(project: Project, path: str | Path) -> None:
    bouts_table(project).to_csv(path, index=False)


def export_summary_csv(project: Project, path: str | Path) -> None:
    s = project.summary()
    with open(path, "w", newline="") as fh:
        s.to_csv(fh, index=False)
        z = project.zone_table()
        if len(z):
            fh.write("\n")
            z.to_csv(fh, index=False)


def export_boris_csv(project: Project, path: str | Path, subject: str = "") -> None:
    fps = project.fps or 30.0
    media = Path(project.video_path).name if project.video_path else ""
    duration = project.n_frames / fps
    rows = []
    for b in project.bouts():
        for status, t in (("START", b["start"] / fps), ("STOP", (b["end"] + 1) / fps)):
            rows.append(
                {
                    "Observation id": Path(media).stem or "observation",
                    "Observation type": "Media file",
                    "Source": "SCRUM",
                    "Media duration (s)": round(duration, 3),
                    "FPS (frame/s)": fps,
                    "Subject": subject,
                    "Behavior": b["behavior"],
                    "Behavior type": "STATE",
                    "Time": round(t, 3),
                    "Media file name": media,
                    "Image index": int(round(t * fps)),
                    "Comment": "edited" if (project.manual[b["start"] : b["end"] + 1] != "").any() else "",
                    "Status": status,
                }
            )
    cols = ["Observation id", "Observation type", "Source", "Media duration (s)", "FPS (frame/s)", "Subject", "Behavior",
            "Behavior type", "Time", "Media file name", "Image index", "Comment", "Status"]
    df = pd.DataFrame(rows, columns=cols).sort_values(["Time", "Status"], ascending=[True, False], kind="stable")
    df.to_csv(path, index=False)


def _frames(project: Project, renderer, start: int, end: int, step: int = 1, max_width: int | None = None):
    cap = cv2.VideoCapture(project.video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open {project.video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for i in range(start, end + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if (i - start) % step:
                continue
            out = renderer.render(frame, i)
            if max_width and out.shape[1] > max_width:
                scale = max_width / out.shape[1]
                out = cv2.resize(out, (max_width, int(out.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            yield i, out
    finally:
        cap.release()


def _stamp(img: np.ndarray, project: Project, frame: int) -> np.ndarray:
    from scrum.app.overlay import label_width, put_label

    text = f"{frame / (project.fps or 30.0):.2f}s  frame {frame}"
    put_label(img, text, (img.shape[1] - label_width(text, 0.5) - 6, img.shape[0] - 14), scale=0.5)
    return img


def export_clip(project: Project, renderer, path: str | Path, start: int, end: int, speed: float = 1.0,
                timestamp: bool = True, progress=None) -> int:
    path = str(path)
    codec = "MJPG" if path.lower().endswith(".avi") else "mp4v"
    writer = None
    n = 0
    total = end - start + 1
    for i, img in _frames(project, renderer, start, end):
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), (project.fps or 30.0) * speed, (w, h))
            if not writer.isOpened():
                raise IOError(f"Could not open a video writer for {path}")
        writer.write(_stamp(img, project, i) if timestamp else img)
        n += 1
        if progress is not None and progress(n, total) is False:
            break
    if writer is not None:
        writer.release()
    return n


def export_gif(project: Project, renderer, path: str | Path, start: int, end: int, max_fps: float = 12.0,
               max_width: int = 480, speed: float = 1.0, progress=None) -> int:
    from PIL import Image

    fps = project.fps or 30.0
    step = max(1, int(round(fps / max_fps)))
    total = (end - start) // step + 1
    images = []
    for k, (i, img) in enumerate(_frames(project, renderer, start, end, step=step, max_width=max_width)):
        images.append(Image.fromarray(cv2.cvtColor(_stamp(img, project, i), cv2.COLOR_BGR2RGB)).convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
        if progress is not None and progress(k + 1, total) is False:
            break
    if not images:
        raise IOError("No frames could be read for the GIF")
    duration_ms = int(round(1000 * step / (fps * speed)))
    images[0].save(str(path), save_all=True, append_images=images[1:], duration=max(20, duration_ms), loop=0, optimize=False, disposal=2)
    return len(images)


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _fig_to_png(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf


def _hex(name: str) -> str:
    r, g, b = behavior_rgb(name)
    return f"#{r:02x}{g:02x}{b:02x}"


def _style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9aa0a6")
    ax.tick_params(colors="#3c4043", labelsize=8)
    ax.grid(axis="y", color="#e8eaed", linewidth=0.6)
    ax.set_axisbelow(True)


def chart_timeline(project: Project):
    plt = _mpl()
    fps = project.fps or 30.0
    fig, ax = plt.subplots(figsize=(7.2, 1.7))
    rows = [("Final", project.final_labels()), ("Model", np.where(project.pred == "", BACKGROUND, project.pred))]
    for yi, (name, labels) in enumerate(rows):
        for b in analysis.bouts_from_labels(labels):
            ax.barh(yi, (b["end"] - b["start"] + 1) / fps, left=b["start"] / fps, height=0.7, color=_hex(b["behavior"]), linewidth=0)
    ax.set_yticks([0, 1], [r[0] for r in rows], fontsize=8)
    ax.set_xlim(0, project.n_frames / fps)
    ax.set_xlabel("time (s)", fontsize=8)
    _style(ax)
    ax.grid(axis="x", color="#e8eaed", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=_hex(b)) for b in project.behaviors() if b != BACKGROUND]
    ax.legend(handles, [b for b in project.behaviors() if b != BACKGROUND], ncol=4, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    png = _fig_to_png(fig)
    plt.close(fig)
    return png


def chart_time_per_behavior(project: Project):
    plt = _mpl()
    s = project.summary()
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.bar(s["behavior"], s["seconds"], color=[_hex(b) for b in s["behavior"]], width=0.6)
    for i, (sec, nb) in enumerate(zip(s["seconds"], s["bouts"])):
        ax.text(i, sec, f"{sec:.1f}s\n{nb} bout{'' if nb == 1 else 's'}", ha="center", va="bottom", fontsize=7, color="#3c4043")
    ax.set_ylabel("seconds", fontsize=8)
    ax.set_ylim(0, max(1.0, s["seconds"].max() * 1.35))
    _style(ax)
    png = _fig_to_png(fig)
    plt.close(fig)
    return png


def chart_traces(project: Project):
    plt = _mpl()
    fps = project.fps or 30.0
    t = np.arange(project.n_frames) / fps
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 2.6), sharex=True)
    palette = ["#d98200", "#0a8fa8", "#2e8b57", "#8e44ad"]
    for i, (a, v) in enumerate(project.speed.items()):
        axes[0].plot(t, v, color=palette[i % 4], linewidth=0.9, label=a)
    axes[0].set_ylabel(f"speed ({project.units}/s)", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False, ncol=2, loc="upper right")
    axes[1].plot(t, project.distance, color="#3c4043", linewidth=0.9)
    axes[1].set_ylabel(f"distance ({project.units})", fontsize=8)
    axes[1].set_xlabel("time (s)", fontsize=8)
    for ax in axes:
        _style(ax)
    png = _fig_to_png(fig)
    plt.close(fig)
    return png


def chart_heatmap(project: Project, background: np.ndarray | None):
    plt = _mpl()
    heat = analysis.occupancy_heatmap(project.positions, (project.height, project.width))
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    if background is not None:
        ax.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB), extent=(0, project.width, project.height, 0))
    im = ax.imshow(np.ma.masked_less(heat, 0.03), extent=(0, project.width, project.height, 0), cmap="inferno", alpha=0.75, vmin=0, vmax=1, interpolation="bilinear")
    for name, poly in project.zones.items():
        pts = np.asarray(poly + [poly[0]])
        ax.plot(pts[:, 0], pts[:, 1], color="#ff5bd8", linewidth=1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label("relative time spent", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    png = _fig_to_png(fig)
    plt.close(fig)
    return png


def chart_transitions(project: Project):
    plt = _mpl()
    behaviors = [b for b in project.behaviors() if b != BACKGROUND]
    counts, probs = analysis.transition_matrix(project.bouts(), behaviors)
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    ax.imshow(probs, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(behaviors)), behaviors, fontsize=7, rotation=30)
    ax.set_yticks(range(len(behaviors)), behaviors, fontsize=7)
    ax.set_xlabel("next behavior", fontsize=8)
    ax.set_ylabel("current behavior", fontsize=8)
    for i in range(len(behaviors)):
        for j in range(len(behaviors)):
            if counts[i, j]:
                ax.text(j, i, f"{probs[i, j]:.0%}\n(n={int(counts[i, j])})", ha="center", va="center", fontsize=7, color="white" if probs[i, j] > 0.5 else "#202124")
    png = _fig_to_png(fig)
    plt.close(fig)
    return png


def export_pdf_report(project: Project, path: str | Path, renderer=None) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    fps = project.fps or 30.0
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=18, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4)
    small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=8.5, textColor=colors.HexColor("#5f6368"))
    body = styles["BodyText"]

    first_frame = None
    if project.video_path and Path(project.video_path).exists():
        cap = cv2.VideoCapture(project.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(project.n_frames - 1, int(project.n_frames * 0.4)))
        ok, fr = cap.read()
        cap.release()
        first_frame = fr if ok else None

    def img(buf, width_in):
        im = Image(buf)
        ratio = im.imageHeight / float(im.imageWidth)
        im.drawWidth, im.drawHeight = width_in * inch, width_in * inch * ratio
        return im

    story = [Paragraph("SCRUM behavior report", h1)]
    name = Path(project.video_path).name if project.video_path else "(no video)"
    story.append(Paragraph(f"{name}", body))
    merged_pct = 100.0 * project.merged.mean() if len(project.merged) else 0.0
    low = analysis.low_confidence_stretches(project.conf, 0.6, 3)
    low_frames = sum(e - s + 1 for s, e in low)
    meta = [
        f"{project.n_frames} frames at {fps:.2f} fps ({project.n_frames / fps:.1f} s)",
        f"labels: {project.source or 'loaded predictions'}",
        f"{project.n_manual_frames()} frames hand-edited",
        f"animals fused into one blob on {merged_pct:.1f}% of frames",
        f"{low_frames} frames in {len(low)} low-confidence stretches (confidence below 0.6)" if not np.all(np.isnan(project.conf)) else "no per-frame confidence available",
        f"scale: {project.px_per_cm:.2f} px per cm (arena calibration)" if project.px_per_cm else "no arena calibration: distances and speeds are in pixels",
        f"generated {datetime.now():%Y-%m-%d %H:%M}",
    ]
    story.append(Paragraph(" | ".join(meta), small))

    s = project.summary()
    story.append(Paragraph("Time per behavior", h2))
    table = [["behavior", "total (s)", "% of video", "bouts", "mean bout (s)", "longest (s)", "latency (s)"]]
    for _, r in s.iterrows():
        table.append([r["behavior"], f"{r['seconds']:.1f}", f"{r['percent']:.1f}", int(r["bouts"]), f"{r['mean_bout_s']:.2f}", f"{r['longest_bout_s']:.2f}",
                      "never" if np.isnan(r["latency_s"]) else f"{r['latency_s']:.1f}"])
    t = Table(table, hAlign="LEFT")
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8.5), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eaed")),
                           ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#bdc1c6")), ("ALIGN", (1, 0), (-1, -1), "RIGHT")]))
    story += [t, Spacer(1, 8), img(chart_timeline(project), 7.0)]

    pair = Table([[img(chart_time_per_behavior(project), 3.4), img(chart_transitions(project), 3.4)]], hAlign="LEFT")
    pair.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(KeepTogether([Paragraph("Totals and sequence", h2), pair]))

    if project.positions:
        story.append(KeepTogether([Paragraph("Movement and position", h2), img(chart_traces(project), 7.0)]))
        story.append(img(chart_heatmap(project, first_frame), 3.6))
    if project.zones:
        z = project.zone_table()
        zt = [["zone", "animal", "time (s)", "% of tracked time"]] + [[r.zone, r.animal, f"{r.seconds:.1f}", f"{r.percent:.1f}"] for r in z.itertuples()]
        story.append(Paragraph("Time in zones", h2))
        zt_table = Table(zt, hAlign="LEFT")
        zt_table.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8.5), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eaed")), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#bdc1c6"))]))
        story.append(zt_table)

    if renderer is not None and project.video_path and Path(project.video_path).exists():
        stills = []
        for beh in [b for b in project.behaviors() if b != BACKGROUND]:
            bouts = [b for b in project.bouts() if b["behavior"] == beh]
            if not bouts:
                continue
            longest = max(bouts, key=lambda b: b["end"] - b["start"])
            f = (longest["start"] + longest["end"]) // 2
            cap = cv2.VideoCapture(project.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            cap.release()
            if ok:
                rendered = renderer.render(frame, f)
                ok2, png = cv2.imencode(".png", cv2.resize(rendered, (480, int(rendered.shape[0] * 480 / rendered.shape[1]))))
                if ok2:
                    stills.append((beh, f, io.BytesIO(png.tobytes())))
        if stills:
            story.append(PageBreak())
            story.append(Paragraph("Examples (middle of the longest bout of each behavior)", h2))
            rows = [[img(b, 3.0), Paragraph(f"<b>{beh}</b><br/>frame {f} ({f / fps:.1f} s)", small)] for beh, f, b in stills]
            story.append(Table(rows, hAlign="LEFT", colWidths=[3.2 * inch, 2.5 * inch]))

    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch, topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                            title="SCRUM behavior report", author="SCRUM")
    doc.build(story)
