"""Draw Target + black_contour bounding boxes on images for visual QA.

Produces annotated images in data/viz/ so you can check whether
annotations align with the actual image content.

Usage:
    python scripts/visualize_annotations.py                # all images
    python scripts/visualize_annotations.py --limit 20     # first 20
    python scripts/visualize_annotations.py --exif         # apply EXIF transpose before drawing
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

DATA_DIR = Path("data")
SRC = DATA_DIR / "instances_clean.json"
IMAGES_DIR = DATA_DIR / "images"
VIZ_DIR = DATA_DIR / "viz"

COLORS = {
    "Target": "lime",
    "black_contour": "cyan",
}
DEFAULT_COLOR = "red"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max images to visualize (0=all)")
    parser.add_argument("--exif", action="store_true", help="Apply EXIF transpose before drawing")
    args = parser.parse_args()

    coco = json.loads(SRC.read_text(encoding="utf-8"))
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_img: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    images = coco["images"]
    if args.limit > 0:
        images = images[: args.limit]

    suffix = "_exif" if args.exif else "_raw"
    for i, img_meta in enumerate(images):
        img_path = IMAGES_DIR / img_meta["file_name"]
        if not img_path.is_file():
            continue

        pil = Image.open(img_path)
        if args.exif:
            pil = ImageOps.exif_transpose(pil)
        pil = pil.convert("RGB")

        draw = ImageDraw.Draw(pil)
        anns = anns_by_img.get(img_meta["id"], [])
        for a in anns:
            x, y, w, h = a["bbox"]
            cat_name = cat_map.get(a["category_id"], "?")
            color = COLORS.get(cat_name, DEFAULT_COLOR)
            draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
            draw.text((x + 4, y + 2), f"{cat_name}", fill=color)

        out_name = Path(img_meta["file_name"]).stem + suffix + ".jpg"
        pil.save(VIZ_DIR / out_name, quality=90)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1} / {len(images)} ...")

    print(f"Done. Wrote {len(images)} annotated images to {VIZ_DIR}/")
    print(f"  Mode: {'EXIF transpose applied' if args.exif else 'raw (no EXIF transpose)'}")


if __name__ == "__main__":
    main()
