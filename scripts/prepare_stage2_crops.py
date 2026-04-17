"""Produce cropped images and data/instances_stage2.json for the bullet detector.

For each Target bbox in the cleaned dataset, crop the original image (with 10% padding),
remap non-Target annotations to crop-relative coordinates, and save.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps

DATA_DIR = Path("data")
SRC = DATA_DIR / "instances_clean.json"
DST = DATA_DIR / "instances_stage2.json"
IMAGES_DIR = DATA_DIR / "images"
CROPS_DIR = DATA_DIR / "crops"
PADDING = 0.05

TARGET_CAT_ID = 12
NON_TARGET_IDS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13}

OLD_TO_NEW = {
    1: 0,   # Bullet_0
    2: 1,   # Bullet_1
    3: 2,   # Bullet_10
    4: 3,   # Bullet_2
    5: 4,   # Bullet_3
    6: 5,   # Bullet_4
    7: 6,   # Bullet_5
    8: 7,   # Bullet_6
    9: 8,   # Bullet_7
    10: 9,  # Bullet_8
    11: 10, # Bullet_9
    13: 11, # black_contour
}

NEW_CATEGORIES = [
    {"id": 0, "name": "Bullet_0", "supercategory": "none"},
    {"id": 1, "name": "Bullet_1", "supercategory": "none"},
    {"id": 2, "name": "Bullet_10", "supercategory": "none"},
    {"id": 3, "name": "Bullet_2", "supercategory": "none"},
    {"id": 4, "name": "Bullet_3", "supercategory": "none"},
    {"id": 5, "name": "Bullet_4", "supercategory": "none"},
    {"id": 6, "name": "Bullet_5", "supercategory": "none"},
    {"id": 7, "name": "Bullet_6", "supercategory": "none"},
    {"id": 8, "name": "Bullet_7", "supercategory": "none"},
    {"id": 9, "name": "Bullet_8", "supercategory": "none"},
    {"id": 10, "name": "Bullet_9", "supercategory": "none"},
    {"id": 11, "name": "black_contour", "supercategory": "none"},
]


def _padded_crop_box(
    tx: float, ty: float, tw: float, th: float,
    img_w: int, img_h: int,
) -> tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) for the padded crop, clamped to image bounds."""
    pad_x = tw * PADDING
    pad_y = th * PADDING
    x1 = max(0, int(tx - pad_x))
    y1 = max(0, int(ty - pad_y))
    x2 = min(img_w, int(tx + tw + pad_x))
    y2 = min(img_h, int(ty + th + pad_y))
    return x1, y1, x2, y2


def _remap_bbox(
    bx: float, by: float, bw: float, bh: float,
    crop_x1: int, crop_y1: int, crop_w: int, crop_h: int,
) -> tuple[float, float, float, float] | None:
    """Remap a bbox to crop-relative coordinates. Returns None if outside crop."""
    new_x = bx - crop_x1
    new_y = by - crop_y1
    new_x2 = new_x + bw
    new_y2 = new_y + bh

    new_x = max(0.0, new_x)
    new_y = max(0.0, new_y)
    new_x2 = min(float(crop_w), new_x2)
    new_y2 = min(float(crop_h), new_y2)

    clipped_w = new_x2 - new_x
    clipped_h = new_y2 - new_y
    if clipped_w < 2 or clipped_h < 2:
        return None
    return (new_x, new_y, clipped_w, clipped_h)


def _to_grayscale_pil(img: Image.Image) -> Image.Image:
    """Convert to grayscale, returned as 3-channel RGB."""
    return img.convert("L").convert("RGB")


def main() -> None:
    if not SRC.is_file():
        raise FileNotFoundError(f"{SRC} not found. Run scripts/data_quality.py first.")
    coco = json.loads(SRC.read_text(encoding="utf-8"))

    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    id_to_img = {img["id"]: img for img in coco["images"]}
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image[a["image_id"]].append(a)

    target_anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for a in coco["annotations"]:
        if a["category_id"] == TARGET_CAT_ID:
            target_anns_by_image[a["image_id"]].append(a)

    new_images: list[dict] = []
    new_annotations: list[dict] = []
    crop_id = 0
    ann_id = 0
    empty_crops = 0

    for img in coco["images"]:
        img_id = img["id"]
        targets = target_anns_by_image.get(img_id, [])
        if not targets:
            continue

        img_path = IMAGES_DIR / img["file_name"]
        if not img_path.is_file():
            print(f"[WARN] Skipping missing image: {img_path}")
            continue

        pil_img = ImageOps.exif_transpose(Image.open(img_path))
        img_w, img_h = pil_img.size

        non_target_anns = [a for a in anns_by_image[img_id]
                           if a["category_id"] in NON_TARGET_IDS]

        for t_idx, t_ann in enumerate(targets):
            tx, ty, tw, th = t_ann["bbox"]
            cx1, cy1, cx2, cy2 = _padded_crop_box(tx, ty, tw, th, img_w, img_h)
            crop_w = cx2 - cx1
            crop_h = cy2 - cy1
            if crop_w < 10 or crop_h < 10:
                continue

            crop_anns = []
            for a in non_target_anns:
                bx, by, bw, bh = a["bbox"]
                remapped = _remap_bbox(bx, by, bw, bh, cx1, cy1, crop_w, crop_h)
                if remapped is None:
                    continue
                crop_anns.append({
                    "id": ann_id,
                    "image_id": crop_id,
                    "category_id": OLD_TO_NEW[a["category_id"]],
                    "bbox": list(remapped),
                    "area": remapped[2] * remapped[3],
                    "iscrowd": 0,
                })
                ann_id += 1

            if not crop_anns:
                empty_crops += 1

            suffix = f"_crop{t_idx}" if len(targets) > 1 else "_crop"
            stem = Path(img["file_name"]).stem
            crop_fname = f"{stem}{suffix}.jpg"
            cropped = pil_img.crop((cx1, cy1, cx2, cy2))
            cropped = _to_grayscale_pil(cropped)
            cropped.save(CROPS_DIR / crop_fname, quality=95)

            new_images.append({
                "id": crop_id,
                "file_name": crop_fname,
                "width": crop_w,
                "height": crop_h,
            })
            new_annotations.extend(crop_anns)
            crop_id += 1

    stage2 = {
        "categories": NEW_CATEGORIES,
        "images": new_images,
        "annotations": new_annotations,
    }
    DST.write_text(json.dumps(stage2), encoding="utf-8")
    print(f"Stage 2 dataset: {len(new_images)} crops, "
          f"{len(new_annotations)} annotations, "
          f"12 categories -> {DST}")
    print(f"  Kept {empty_crops} crops with zero bullet annotations.")
    print(f"  Crop images saved to {CROPS_DIR}/")


if __name__ == "__main__":
    main()
