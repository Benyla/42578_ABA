"""Audit dataset integrity: categories, annotation counts, image alignment."""
import json
from collections import Counter
from pathlib import Path

d = json.loads(Path("data/instances.json").read_text(encoding="utf-8"))

cat_ids = sorted(c["id"] for c in d["categories"])
print(f"Category IDs: {cat_ids}")
print(f"Num categories: {len(d['categories'])}")

cat_counts = Counter(a["category_id"] for a in d["annotations"])
for c in d["categories"]:
    name = c["name"]
    count = cat_counts.get(c["id"], 0)
    print(f"  {name} (id={c['id']}): {count} annotations")

disk_imgs = {f.name for f in Path("data/images").iterdir()
             if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}}
json_imgs = {img["file_name"] for img in d["images"]}

print(f"\nImages on disk: {len(disk_imgs)}")
print(f"Images in JSON: {len(json_imgs)}")
missing = json_imgs - disk_imgs
extra = disk_imgs - json_imgs
if missing:
    print(f"  IN JSON BUT NOT ON DISK ({len(missing)}): {list(missing)[:5]}")
if extra:
    print(f"  ON DISK BUT NOT IN JSON ({len(extra)}): {list(extra)[:5]}")
if not missing and not extra:
    print("  Perfect match.")
