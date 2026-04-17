"""Audit and clean dataset. Produces data/instances_clean.json."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path("data")
SRC = DATA_DIR / "instances.json"
DST = DATA_DIR / "instances_clean.json"
IMAGES_DIR = DATA_DIR / "images"

PHANTOM_CAT_NAME = "Shooting-Discs"
TARGET_CAT_NAME = "Target"
BLACK_CONTOUR_CAT_NAME = "black_contour"


def main() -> None:
    coco = json.loads(SRC.read_text(encoding="utf-8"))
    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    print(f"Source: {SRC}")
    print(f"Images: {len(images)}  Annotations: {len(annotations)}  Categories: {len(categories)}")

    # Resolve category ids by name (robust to id changes)
    name_to_cat = {c["name"]: c for c in categories}
    if TARGET_CAT_NAME not in name_to_cat:
        raise SystemExit(f"Missing category named {TARGET_CAT_NAME!r} in {SRC}")
    if BLACK_CONTOUR_CAT_NAME not in name_to_cat:
        raise SystemExit(f"Missing category named {BLACK_CONTOUR_CAT_NAME!r} in {SRC}")
    target_cat_id = int(name_to_cat[TARGET_CAT_NAME]["id"])
    black_contour_cat_id = int(name_to_cat[BLACK_CONTOUR_CAT_NAME]["id"])

    # --- Check 1: bbox type coercion ---
    str_bbox_count = 0
    for ann in annotations:
        coerced = [float(v) for v in ann["bbox"]]
        if coerced != ann["bbox"]:
            str_bbox_count += 1
        ann["bbox"] = coerced
    if str_bbox_count:
        print(f"\n[FIX] Coerced {str_bbox_count} annotations with string bbox values to float.")
    else:
        print("\n[OK] All bbox values are numeric.")

    # --- Check 2: image files on disk ---
    disk_files = {f.name for f in IMAGES_DIR.iterdir()
                  if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}}
    json_files = {img["file_name"] for img in images}
    missing = json_files - disk_files
    if missing:
        print(f"\n[WARN] {len(missing)} images in JSON but not on disk: {list(missing)[:5]}")
    else:
        print(f"[OK] All {len(json_files)} image files exist on disk.")

    # --- Check 3: keep only images with exactly 1 Target and 1 black_contour ---
    id_to_img = {img["id"]: img for img in images}
    per_img_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for a in annotations:
        per_img_counts[int(a["image_id"])][int(a["category_id"])] += 1

    keep_ids: set[int] = set()
    drop_ids: set[int] = set()
    for img in images:
        img_id = int(img["id"])
        counts = per_img_counts.get(img_id, {})
        t = int(counts.get(target_cat_id, 0))
        bc = int(counts.get(black_contour_cat_id, 0))
        if t == 1 and bc == 1:
            keep_ids.add(img_id)
        else:
            drop_ids.add(img_id)

    if drop_ids:
        # Provide a small breakdown for transparency
        missing_target = 0
        missing_bc = 0
        multi_target = 0
        multi_bc = 0
        other = 0
        dropped_missing_target: list[str] = []
        dropped_multi_target: list[str] = []
        dropped_missing_bc: list[str] = []
        dropped_multi_bc: list[str] = []
        dropped_other: list[str] = []
        for img_id in drop_ids:
            counts = per_img_counts.get(img_id, {})
            t = int(counts.get(target_cat_id, 0))
            bc = int(counts.get(black_contour_cat_id, 0))
            if t == 0:
                missing_target += 1
                dropped_missing_target.append(id_to_img[img_id]["file_name"])
            elif t > 1:
                multi_target += 1
                dropped_multi_target.append(id_to_img[img_id]["file_name"])
            if bc == 0:
                missing_bc += 1
                dropped_missing_bc.append(id_to_img[img_id]["file_name"])
            elif bc > 1:
                multi_bc += 1
                dropped_multi_bc.append(id_to_img[img_id]["file_name"])
            if not (t == 0 or t > 1 or bc == 0 or bc > 1):
                other += 1
                dropped_other.append(id_to_img[img_id]["file_name"])

        print(
            f"\n[FIX] Keeping only images with exactly 1 {TARGET_CAT_NAME} and 1 {BLACK_CONTOUR_CAT_NAME}.\n"
            f"      Dropped {len(drop_ids)} / {len(images)} images.\n"
            f"      missing_target={missing_target}\n"
            f"      multi_target={multi_target}\n"
            f"      missing_black_contour={missing_bc}\n"
            f"      multi_black_contour={multi_bc}\n"
            f"      other={other}"
        )

        def _print_list(title: str, items: list[str], limit: int = 200) -> None:
            if not items:
                return
            print(f"\n      {title} ({len(items)}):")
            for name in sorted(items)[:limit]:
                print(f"        - {name}")
            if len(items) > limit:
                print(f"        ... ({len(items) - limit} more)")

        _print_list("Dropped (missing Target)", dropped_missing_target)
        _print_list("Dropped (multiple Targets)", dropped_multi_target)
        _print_list("Dropped (missing black_contour)", dropped_missing_bc)
        _print_list("Dropped (multiple black_contour)", dropped_multi_bc)
        _print_list("Dropped (other)", dropped_other)

        images = [img for img in images if int(img["id"]) in keep_ids]
        annotations = [a for a in annotations if int(a["image_id"]) in keep_ids]
    else:
        print(f"[OK] All images have exactly 1 {TARGET_CAT_NAME} and 1 {BLACK_CONTOUR_CAT_NAME}.")

    # --- Check 5: drop phantom category ---
    phantom = [c for c in categories if c["name"] == PHANTOM_CAT_NAME]
    cat_ann_counts = Counter(a["category_id"] for a in annotations)
    removed_cats = []
    for c in phantom:
        if cat_ann_counts.get(c["id"], 0) == 0:
            removed_cats.append(c)
    if removed_cats:
        drop_cat_ids = {c["id"] for c in removed_cats}
        categories = [c for c in categories if c["id"] not in drop_cat_ids]
        print(f"\n[FIX] Dropped phantom categories with 0 annotations: "
              f"{[c['name'] for c in removed_cats]}")

    # --- Summary ---
    cat_counts = Counter(a["category_id"] for a in annotations)
    cat_name_map = {c["id"]: c["name"] for c in categories}
    print(f"\n{'=' * 60}")
    print("CLEANED DATASET SUMMARY")
    print(f"{'=' * 60}")
    print(f"Images: {len(images)}")
    print(f"Annotations: {len(annotations)}")
    print(f"Categories: {len(categories)}")
    print("\nAnnotation distribution:")
    for c in categories:
        print(f"  {c['name']} (id={c['id']}): {cat_counts.get(c['id'], 0)}")

    # Explicit summary block (easy to copy/paste)
    print("\nProduced data/instances_clean.json with:")
    print(f"{len(images)} images")
    print(f"{cat_counts.get(target_cat_id, 0)} {TARGET_CAT_NAME} annotations")
    print(f"{cat_counts.get(black_contour_cat_id, 0)} {BLACK_CONTOUR_CAT_NAME} annotations")

    # --- Write ---
    clean = dict(coco)
    clean["images"] = images
    clean["annotations"] = annotations
    clean["categories"] = categories
    DST.write_text(json.dumps(clean), encoding="utf-8")
    print(f"\nWrote {DST}")


if __name__ == "__main__":
    main()
