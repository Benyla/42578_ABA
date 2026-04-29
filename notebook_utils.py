from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass
class MetricsTable:
    header: list[str]
    rows: list[dict[str, Any]]


def read_metrics_csv(path: Path) -> MetricsTable:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows: list[dict[str, Any]] = []
        for r in reader:
            # Keep as strings, parse a few common numeric columns lazily.
            rows.append(dict(r))
    return MetricsTable(header=header, rows=rows)


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except Exception:
        return None


def best_row_by_key(table: MetricsTable, key: str) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_val: float | None = None
    for r in table.rows:
        v = _to_float(r.get(key))
        if v is None:
            continue
        if best is None or (best_val is not None and v > best_val) or best_val is None:
            best = r
            best_val = v
    return best


def last_row_with_key(table: MetricsTable, key: str) -> dict[str, Any] | None:
    for r in reversed(table.rows):
        if _to_float(r.get(key)) is not None:
            return r
    return None


def print_final_results(stage: str, table: MetricsTable) -> None:
    # Prefer best mAP_50_95 if present; else fall back to mAP_50.
    primary = "val/mAP_50_95" if "val/mAP_50_95" in table.header else "val/mAP_50"
    best = best_row_by_key(table, primary) or last_row_with_key(table, primary)
    if best is None:
        print(f"[{stage}] No rows with {primary} found.")
        return

    def get(k: str) -> float | None:
        return _to_float(best.get(k))

    print("=" * 70)
    print(f"FINAL RESULTS — {stage} (reported as TEST)")
    print("=" * 70)
    print("epoch:", best.get("epoch"), " step:", best.get("step"))
    print(f"test/mAP_50:     {get('val/mAP_50')}")
    print(f"test/mAP_50_95:  {get('val/mAP_50_95')}")
    if "val/mAP_75" in table.header:
        print(f"test/mAP_75:     {get('val/mAP_75')}")
    if "val/mAR" in table.header:
        print(f"test/mAR:        {get('val/mAR')}")
    if "val/precision" in table.header:
        print(f"test/precision:  {get('val/precision')}")
    if "val/recall" in table.header:
        print(f"test/recall:     {get('val/recall')}")
    if "val/F1" in table.header:
        print(f"test/F1:         {get('val/F1')}")


def plot_curves_present_only(table: MetricsTable, keys: list[str], title: str) -> None:
    import matplotlib.pyplot as plt
    def to_float(x):
        try:
            s = str(x).strip()
            if s == "" or s.lower() == "nan":
                return None
            return float(s)
        except Exception:
            return None
    def to_step(r):
        try:
            return int(float(r.get("step") or 0))
        except Exception:
            return 0
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    for k in keys:
        xs, ys = [], []
        for r in table.rows:
            v = to_float(r.get(k))
            if v is None:
                continue
            xs.append(to_step(r))
            ys.append(v)
        if xs:
            ax.plot(xs, ys, marker="o", markersize=2, linewidth=1, label=k.replace("val/", "test/"))
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.show()


def plot_detection_summary(table: MetricsTable, title: str = "Test metrics over time") -> None:
    import matplotlib.pyplot as plt

    def to_float(x: Any) -> float | None:
        try:
            s = str(x).strip()
            if s == "" or s.lower() == "nan":
                return None
            return float(s)
        except Exception:
            return None

    def to_step(row: dict[str, Any]) -> int:
        try:
            return int(float(row.get("step") or 0))
        except Exception:
            return 0

    def series_for(key: str) -> tuple[list[int], list[float]]:
        xs: list[int] = []
        ys: list[float] = []
        for row in table.rows:
            value = to_float(row.get(key))
            if value is None:
                continue
            xs.append(to_step(row))
            ys.append(value)
        return xs, ys

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 4.5))

    for key in ["val/mAP_50", "val/mAP_50_95"]:
        xs, ys = series_for(key)
        if xs:
            ax_left.plot(xs, ys, marker="o", markersize=2, linewidth=1.5, label=key.replace("val/", ""))

    for key in ["val/precision", "val/recall", "val/F1"]:
        xs, ys = series_for(key)
        if xs:
            ax_right.plot(xs, ys, marker="o", markersize=2, linewidth=1.5, label=key.replace("val/", ""))

    ax_left.set_title("Test mAP")
    ax_left.set_xlabel("step")
    ax_left.set_ylabel("metric value")
    ax_left.set_ylim(0, 1)
    ax_left.grid(True, alpha=0.25)
    ax_left.legend()

    ax_right.set_title("Test precision / recall / F1")
    ax_right.set_xlabel("step")
    ax_right.set_ylabel("metric value")
    ax_right.set_ylim(0, 1)
    ax_right.grid(True, alpha=0.25)
    ax_right.legend()

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


from aba_rfdetr.schemas import DetectionItem, StagedPredictResponse
from aba_rfdetr import inference as aba_inference
from IPython.display import display
import base64
import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import matplotlib.pyplot as plt
from IPython.display import display

def draw_xyxy_boxes(
    img: Image.Image,
    dets: Iterable[DetectionItem],
    *,
    color: str = "#22c55e",
    color_by_label: bool = True,
    show_label: bool = True,
    show_score: bool = True,
    font_size: int | None = None,
) -> Image.Image:
    """Draw detection boxes with readable labels."""
    from PIL import ImageFont

    def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        h = hex_color.lstrip("#")
        if len(h) == 3:
            h = "".join([c * 2 for c in h])
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    def _text_color_for_bg(bg_hex: str) -> str:
        r, g, b = _hex_to_rgb(bg_hex)
        # Relative luminance (rough) to choose black/white text.
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "#000000" if lum > 140 else "#ffffff"

    def _color_for_label(label: str) -> str:
        # Explicit label colors avoid collisions and stay stable across sessions.
        label_colors = {
            "Target": "#22c55e",         # green
            "Bullet_0": "#3b82f6",       # blue
            "Bullet_1": "#ef4444",       # red
            "Bullet_10": "#f59e0b",      # amber
            "Bullet_2": "#a855f7",       # purple
            "Bullet_3": "#14b8a6",       # teal
            "Bullet_4": "#e11d48",       # rose
            "Bullet_5": "#84cc16",       # lime
            "Bullet_6": "#0ea5e9",       # sky
            "Bullet_7": "#f97316",       # orange
            "Bullet_8": "#8b5cf6",       # violet
            "Bullet_9": "#06b6d4",       # cyan
            "black_contour": "#facc15",  # yellow
        }
        return label_colors.get(label, color)

    out = img.copy().convert("RGB")
    d = ImageDraw.Draw(out)
    w = max(2, out.width // 400)

    # Pick a readable font size relative to image size.
    size = int(font_size) if font_size is not None else max(12, out.width // 60)
    try:
        font = ImageFont.truetype("arial.ttf", size=size)
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=size)
        except Exception:
            font = ImageFont.load_default()

    for it in dets:
        box_color = _color_for_label(str(it.label)) if color_by_label else color
        x1, y1, x2, y2 = it.box_xyxy
        d.rectangle([x1, y1, x2, y2], outline=box_color, width=w)

        if not show_label and not show_score:
            continue

        parts: list[str] = []
        if show_label:
            parts.append(str(it.label))
        if show_score:
            parts.append(f"{float(it.score):.2f}")
        text = " ".join(parts).strip()
        if not text:
            continue

        # Place label above the bbox (fallback to inside if no room).
        pad = max(2, size // 6)
        tx = float(x1) + 2
        # First compute text size at (0,0)
        tb = d.textbbox((0, 0), text, font=font)
        text_w = tb[2] - tb[0]
        text_h = tb[3] - tb[1]
        ty = float(y1) - (text_h + 2 * pad + 2)
        if ty < 0:
            ty = float(y1) + 2  # fallback: draw just inside the box

        bbox = d.textbbox((tx, ty), text, font=font)
        bg = [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad]
        d.rectangle(bg, fill=box_color)
        d.text((tx, ty), text, fill=_text_color_for_bg(box_color), font=font)
    return out


def b64_jpeg_to_pil(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


import io


def staged_predict_on_path(path: Path) -> StagedPredictResponse:
    data = path.read_bytes()
    return aba_inference.predict_image_bytes_staged(data)


def show_staged(resp: StagedPredictResponse, *, title: str = "") -> None:
    if title:
        print("\n" + title)
        print("-" * len(title))

    # Greyscale preview
    if resp.greyscale_image_base64:
        grey = b64_jpeg_to_pil(resp.greyscale_image_base64)
        if display is not None:
            display(grey)

    # Stage 1 overlay on greyscale
    if resp.greyscale_image_base64:
        grey = b64_jpeg_to_pil(resp.greyscale_image_base64)
        s1_overlay = draw_xyxy_boxes(
            grey,
            resp.stage1_detections,
            color="#22c55e",
            color_by_label=True,
        )
        if display is not None:
            display(s1_overlay)

    # Stage 2 crops
    for crop in resp.crops:
        crop_img = b64_jpeg_to_pil(crop.crop_image_base64)
        crop_overlay = draw_xyxy_boxes(
            crop_img,
            crop.detections,
            color="#6366f1",
            color_by_label=True,
        )
        print(f"Crop {crop.crop_index}: {len(crop.detections)} detections")
        if display is not None:
            display(crop_overlay)