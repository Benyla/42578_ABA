"""Prepare a cached dataset for ResNet training: target_type {1,2}.

Reads:
  - data/target_labels.csv (filename,target_type)
  - data/instances.json (COCO, uses Target bboxes)
  - data/images/<filename>

Writes:
  - data/resnet_type12/train/{1,2}/*.jpg
  - data/resnet_type12/val/{1,2}/*.jpg
  - data/resnet_type12/manifest.csv

Cropping:
  - Uses ground-truth COCO bbox for category name 'Target' (expects 1-2 per image).
  - If 2 targets exist, uses the largest-area box.
  - Optional padding is applied as a fraction of bbox width/height, then clamped.

Greyscale:
  - Matches inference: convert to L then back to RGB (3-channel greyscale).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS = PROJECT_ROOT / "data" / "target_labels.csv"
DEFAULT_COCO = PROJECT_ROOT / "data" / "instances.json"
DEFAULT_IMAGES = PROJECT_ROOT / "data" / "images"
DEFAULT_OUT = PROJECT_ROOT / "data" / "resnet_type12"


def _to_greyscale_rgb(img: Image.Image) -> Image.Image:
    return img.convert("L").convert("RGB")


def _padded_crop_box(x1: float, y1: float, x2: float, y2: float, *, img_w: int, img_h: int, padding: float):
    w = x2 - x1
    h = y2 - y1
    pad_x = w * padding
    pad_y = h * padding
    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(img_w, int(x2 + pad_x))
    cy2 = min(img_h, int(y2 + pad_y))
    return cx1, cy1, cx2, cy2


@dataclass(frozen=True)
class CocoIndex:
    image_id_by_name: dict[str, int]
    anns_by_image_id: dict[int, list[dict]]
    cat_id_by_name_lower: dict[str, int]


def _load_coco_index(path: Path) -> CocoIndex:
    coco = json.loads(path.read_text(encoding="utf-8"))
    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    cats = coco.get("categories", [])

    image_id_by_name: dict[str, int] = {}
    for im in images:
        if isinstance(im, dict) and "file_name" in im and "id" in im:
            image_id_by_name[str(im["file_name"])] = int(im["id"])

    anns_by_image_id: dict[int, list[dict]] = {}
    for a in anns:
        if not isinstance(a, dict):
            continue
        iid = a.get("image_id")
        if iid is None:
            continue
        iid = int(iid)
        anns_by_image_id.setdefault(iid, []).append(a)

    cat_id_by_name_lower: dict[str, int] = {}
    for c in cats:
        if not isinstance(c, dict):
            continue
        if "id" not in c or "name" not in c:
            continue
        cat_id_by_name_lower[str(c["name"]).strip().lower()] = int(c["id"])

    return CocoIndex(
        image_id_by_name=image_id_by_name,
        anns_by_image_id=anns_by_image_id,
        cat_id_by_name_lower=cat_id_by_name_lower,
    )


def _read_labels(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            fn = str(row["filename"])
            t = int(row["target_type"])
            if t in (1, 2):
                rows.append((fn, t))
    return rows


def _choose_target_bbox_xyxy(annotations: list[dict], *, target_cat_id: int) -> tuple[float, float, float, float] | None:
    """Return largest Target bbox in xyxy pixels, or None."""
    best = None
    best_area = -1.0
    for a in annotations:
        if a.get("category_id") != target_cat_id:
            continue
        bb = a.get("bbox")
        if not (isinstance(bb, list) and len(bb) == 4):
            continue
        x, y, w, h = map(float, bb)  # COCO xywh
        area = w * h
        if area > best_area:
            best_area = area
            best = (x, y, x + w, y + h)
    return best


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare cached greyscale crops for ResNet type 1 vs 2")
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--coco", type=Path, default=DEFAULT_COCO)
    p.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--padding", type=float, default=0.05, help="Crop padding fraction around GT bbox.")
    p.add_argument("--max-per-class", type=int, default=0, help="If >0, cap examples per class (for quick runs).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output folder contents.")
    args = p.parse_args()

    labels = _read_labels(args.labels)
    rng = random.Random(args.seed)
    rng.shuffle(labels)

    idx = _load_coco_index(args.coco)
    target_cat_id = idx.cat_id_by_name_lower.get("target")
    if target_cat_id is None:
        raise SystemExit("Could not find category named 'Target' in COCO categories.")

    out_root = args.out
    if args.overwrite and out_root.exists():
        # Safe delete: only remove our expected subfolders.
        for sub in ("train", "val", "manifest.csv"):
            path = out_root / sub
            if path.is_dir():
                for f in path.rglob("*"):
                    if f.is_file():
                        f.unlink()
                for d in sorted([d for d in path.rglob("*") if d.is_dir()], reverse=True):
                    d.rmdir()
                path.rmdir()
            elif path.is_file():
                path.unlink()

    (out_root / "train" / "1").mkdir(parents=True, exist_ok=True)
    (out_root / "train" / "2").mkdir(parents=True, exist_ok=True)
    (out_root / "val" / "1").mkdir(parents=True, exist_ok=True)
    (out_root / "val" / "2").mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    n_val = int(len(labels) * args.val_fraction)
    val_set = set(fn for fn, _ in labels[:n_val])

    per_class_written = {1: 0, 2: 0}
    skipped = 0
    written = 0

    with manifest_path.open("w", newline="", encoding="utf-8") as mf:
        w = csv.DictWriter(
            mf,
            fieldnames=[
                "split",
                "label",
                "filename",
                "output_relpath",
                "bbox_xyxy",
                "padding",
                "skip_reason",
            ],
        )
        w.writeheader()

        for fn, label in labels:
            if args.max_per_class > 0 and per_class_written[label] >= args.max_per_class:
                continue

            split = "val" if fn in val_set else "train"
            img_path = args.images / fn
            if not img_path.exists():
                skipped += 1
                w.writerow(
                    {
                        "split": split,
                        "label": label,
                        "filename": fn,
                        "output_relpath": "",
                        "bbox_xyxy": "",
                        "padding": args.padding,
                        "skip_reason": "missing_image_on_disk",
                    }
                )
                continue

            iid = idx.image_id_by_name.get(fn)
            if iid is None:
                skipped += 1
                w.writerow(
                    {
                        "split": split,
                        "label": label,
                        "filename": fn,
                        "output_relpath": "",
                        "bbox_xyxy": "",
                        "padding": args.padding,
                        "skip_reason": "filename_not_in_coco_images",
                    }
                )
                continue

            anns = idx.anns_by_image_id.get(iid, [])
            bbox = _choose_target_bbox_xyxy(anns, target_cat_id=target_cat_id)
            if bbox is None:
                skipped += 1
                w.writerow(
                    {
                        "split": split,
                        "label": label,
                        "filename": fn,
                        "output_relpath": "",
                        "bbox_xyxy": "",
                        "padding": args.padding,
                        "skip_reason": "no_target_bbox",
                    }
                )
                continue

            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img).convert("RGB")
            img_w, img_h = img.size
            x1, y1, x2, y2 = bbox
            cx1, cy1, cx2, cy2 = _padded_crop_box(x1, y1, x2, y2, img_w=img_w, img_h=img_h, padding=float(args.padding))
            if cx2 <= cx1 or cy2 <= cy1:
                skipped += 1
                w.writerow(
                    {
                        "split": split,
                        "label": label,
                        "filename": fn,
                        "output_relpath": "",
                        "bbox_xyxy": f"{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}",
                        "padding": args.padding,
                        "skip_reason": "invalid_crop_box",
                    }
                )
                continue

            crop = _to_greyscale_rgb(img.crop((cx1, cy1, cx2, cy2)))
            out_rel = Path(split) / str(label) / fn
            out_path = out_root / out_rel
            crop.save(out_path, format="JPEG", quality=95)

            per_class_written[label] += 1
            written += 1
            w.writerow(
                {
                    "split": split,
                    "label": label,
                    "filename": fn,
                    "output_relpath": out_rel.as_posix(),
                    "bbox_xyxy": f"{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}",
                    "padding": args.padding,
                    "skip_reason": "",
                }
            )

    print(f"Prepared dataset at: {out_root}")
    print(f"Written: {written}  Skipped: {skipped}")
    print(f"Per-class written: {per_class_written}")


if __name__ == '__main__':
    main()

