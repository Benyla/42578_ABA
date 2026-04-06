"""One-shot fix: coerce all bbox values in instances.json from str to float."""
import json
from pathlib import Path

path = Path("data/instances.json")
data = json.loads(path.read_text(encoding="utf-8"))

fixed = 0
for ann in data["annotations"]:
    bbox = ann["bbox"]
    new_bbox = [float(v) for v in bbox]
    if new_bbox != bbox:
        fixed += 1
    ann["bbox"] = new_bbox

path.write_text(json.dumps(data), encoding="utf-8")
total = len(data["annotations"])
print(f"Fixed {fixed} / {total} annotations with string bbox values")
