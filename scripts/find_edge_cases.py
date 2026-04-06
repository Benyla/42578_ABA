"""Find images with 2 Target bboxes or bullets but no Target."""
import json
from collections import defaultdict
from pathlib import Path

d = json.loads(Path("data/instances.json").read_text(encoding="utf-8"))
id_to_fname = {img["id"]: img["file_name"] for img in d["images"]}

target_per_img = defaultdict(list)
bullet_imgs = set()
target_imgs = set()

for a in d["annotations"]:
    if a["category_id"] == 12:
        target_per_img[a["image_id"]].append(a)
        target_imgs.add(a["image_id"])
    elif 1 <= a["category_id"] <= 11:
        bullet_imgs.add(a["image_id"])

print("=== Images with 2 Target bboxes ===")
for img_id, anns in sorted(target_per_img.items()):
    if len(anns) == 2:
        boxes = []
        for a in anns:
            b = a["bbox"]
            boxes.append(f"{b[2]:.0f}x{b[3]:.0f}")
        print(f"  {id_to_fname[img_id]}  (boxes: {boxes[0]}, {boxes[1]})")

print(f"\n=== Images with bullets but NO Target ===")
no_target = bullet_imgs - target_imgs
for img_id in sorted(no_target):
    n = sum(1 for a in d["annotations"]
            if a["image_id"] == img_id and 1 <= a["category_id"] <= 11)
    print(f"  {id_to_fname[img_id]}  ({n} bullet annotations)")
