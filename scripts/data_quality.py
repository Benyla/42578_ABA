"""Audit and clean dataset. Produces data/instances_clean.json."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path("data")
SRC = DATA_DIR / "instances.json"
DST = DATA_DIR / "instances_clean.json"
IMAGES_DIR = DATA_DIR / "images"

TARGET_CAT_ID = 12
PHANTOM_CAT_NAME = "Shooting-Discs"


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

    # --- Check 3: images without Target annotation ---
    target_imgs = {a["image_id"] for a in annotations if a["category_id"] == TARGET_CAT_ID}
    id_to_img = {img["id"]: img for img in images}
    no_target = [img for img in images if img["id"] not in target_imgs]
    if no_target:
        print(f"\n[FIX] Removing {len(no_target)} images with no Target annotation:")
        for img in no_target:
            print(f"       {img['file_name']}")
        drop_ids = {img["id"] for img in no_target}
        images = [img for img in images if img["id"] not in drop_ids]
        annotations = [a for a in annotations if a["image_id"] not in drop_ids]
    else:
        print("[OK] Every image has at least one Target annotation.")

    # --- Check 4: images with multiple Targets ---
    target_counts: dict[int, int] = defaultdict(int)
    for a in annotations:
        if a["category_id"] == TARGET_CAT_ID:
            target_counts[a["image_id"]] += 1
    multi = {img_id: cnt for img_id, cnt in target_counts.items() if cnt > 1}
    if multi:
        print(f"\n[INFO] {len(multi)} images have multiple Target bboxes (kept as-is):")
        for img_id, cnt in sorted(multi.items()):
            print(f"       {id_to_img[img_id]['file_name']}  ({cnt} targets)")

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

    # --- Write ---
    clean = dict(coco)
    clean["images"] = images
    clean["annotations"] = annotations
    clean["categories"] = categories
    DST.write_text(json.dumps(clean), encoding="utf-8")
    print(f"\nWrote {DST}")


if __name__ == "__main__":
    main()
