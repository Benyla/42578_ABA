"""Sample 25 random images from each stage/split and draw annotations for visual QA.

Produces a grid of annotated images under data/viz_splits/ so you can quickly
verify that bounding boxes line up with the actual image content.

Usage:
    uv run python scripts/visualize_splits.py
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

N_SAMPLES = 25
COLS = 5
SEED = 123
OUTPUT_DIR = Path("data/viz_splits")

PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabebe",
    "#469990", "#e6beff", "#9A6324", "#800000",
]

SPLITS = {
    "stage1_train": Path("data/rf_train_stage1/train"),
    "stage1_valid": Path("data/rf_train_stage1/valid"),
    "stage2_train": Path("data/rf_train_stage2/train"),
    "stage2_valid": Path("data/rf_train_stage2/valid"),
}


def _load_split(split_dir: Path) -> tuple[dict, dict[int, list[dict]], dict[int, str]]:
    ann_file = split_dir / "_annotations.coco.json"
    coco = json.loads(ann_file.read_text(encoding="utf-8"))
    id_to_img = {img["id"]: img for img in coco["images"]}
    anns_by_img: dict[int, list[dict]] = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    return id_to_img, anns_by_img, cat_names


def _draw_annotations(
    img: Image.Image,
    anns: list[dict],
    cat_names: dict[int, str],
) -> Image.Image:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=max(12, img.width // 40))
    except OSError:
        font = ImageFont.load_default()

    for ann in anns:
        x, y, w, h = ann["bbox"]
        cat_id = ann["category_id"]
        color = PALETTE[cat_id % len(PALETTE)]
        label = cat_names.get(cat_id, str(cat_id))

        draw.rectangle([x, y, x + w, y + h], outline=color, width=max(2, img.width // 300))
        draw.text((x + 2, y + 2), label, fill=color, font=font)

    return img


def _build_grid(images: list[Image.Image], cols: int, thumb_w: int = 400) -> Image.Image:
    thumbs = []
    for im in images:
        ratio = thumb_w / im.width
        thumb = im.resize((thumb_w, int(im.height * ratio)), Image.LANCZOS)
        thumbs.append(thumb)

    max_h = max(t.height for t in thumbs)
    rows = (len(thumbs) + cols - 1) // cols
    grid = Image.new("RGB", (cols * thumb_w, rows * max_h), (30, 30, 30))

    for idx, thumb in enumerate(thumbs):
        r, c = divmod(idx, cols)
        grid.paste(thumb, (c * thumb_w, r * max_h))

    return grid


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    for name, split_dir in SPLITS.items():
        if not split_dir.is_dir():
            print(f"[SKIP] {split_dir} does not exist")
            continue

        id_to_img, anns_by_img, cat_names = _load_split(split_dir)
        img_ids = list(id_to_img.keys())
        sample_ids = rng.sample(img_ids, min(N_SAMPLES, len(img_ids)))

        annotated: list[Image.Image] = []
        for img_id in sample_ids:
            entry = id_to_img[img_id]
            img_path = split_dir / entry["file_name"]
            if not img_path.is_file():
                print(f"  [WARN] Missing: {img_path}")
                continue
            img = Image.open(img_path).convert("RGB")
            anns = anns_by_img.get(img_id, [])
            annotated.append(_draw_annotations(img, anns, cat_names))

        if not annotated:
            print(f"[SKIP] No images found for {name}")
            continue

        grid = _build_grid(annotated, COLS)
        out_path = OUTPUT_DIR / f"{name}.jpg"
        grid.save(out_path, quality=90)
        print(f"[OK] {name}: {len(annotated)} images -> {out_path}")


if __name__ == "__main__":
    main()
