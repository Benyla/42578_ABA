"""One-off: draw annotations on a single image, both raw and EXIF-corrected."""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

DATA_DIR = Path("data")
coco = json.loads((DATA_DIR / "instances_clean.json").read_text(encoding="utf-8"))
cat_map = {c["id"]: c["name"] for c in coco["categories"]}
IMG_ID = 590
anns = [a for a in coco["annotations"] if a["image_id"] == IMG_ID]

COLORS = {"Target": "lime", "black_contour": "cyan"}
fname = "20200923_130815_jpg.rf.9JHyDFGWyaFvWhbahtV5.jpg"
img_path = DATA_DIR / "images" / fname
VIZ = DATA_DIR / "viz"
VIZ.mkdir(exist_ok=True)

for mode in ["raw", "exif"]:
    pil = Image.open(img_path)
    if mode == "exif":
        pil = ImageOps.exif_transpose(pil)
    pil = pil.convert("RGB")
    draw = ImageDraw.Draw(pil)
    for a in anns:
        x, y, w, h = a["bbox"]
        name = cat_map.get(a["category_id"], "?")
        color = COLORS.get(name, "red")
        draw.rectangle([x, y, x + w, y + h], outline=color, width=5)
        draw.text((x + 6, y + 4), name, fill=color)
    out = VIZ / f"{Path(fname).stem}_{mode}.jpg"
    pil.save(out, quality=90)
    print(f"{mode}: size={pil.size} -> {out}")

print(f"\nAnnotations for image {IMG_ID}:")
for a in anns:
    cid = a["category_id"]
    print(f"  {cat_map[cid]}: bbox={a['bbox']}")
