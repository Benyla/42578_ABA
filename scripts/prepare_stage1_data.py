"""Produce data/instances_stage1.json: Target-only annotations remapped to a single class."""
from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path("data")
SRC = DATA_DIR / "instances_clean.json"
DST = DATA_DIR / "instances_stage1.json"

TARGET_CAT_ID = 12


def main() -> None:
    if not SRC.is_file():
        raise FileNotFoundError(
            f"{SRC} not found. Run scripts/data_quality.py first."
        )
    coco = json.loads(SRC.read_text(encoding="utf-8"))

    target_anns = [a for a in coco["annotations"] if a["category_id"] == TARGET_CAT_ID]
    for i, ann in enumerate(target_anns):
        ann["category_id"] = 0
        ann["id"] = i

    stage1 = {
        k: coco[k] for k in ("info", "licenses") if k in coco
    }
    stage1["categories"] = [{"id": 0, "name": "Target", "supercategory": "none"}]
    stage1["images"] = coco["images"]
    stage1["annotations"] = target_anns

    DST.write_text(json.dumps(stage1), encoding="utf-8")
    print(f"Stage 1 dataset: {len(stage1['images'])} images, "
          f"{len(target_anns)} Target annotations -> {DST}")


if __name__ == "__main__":
    main()
