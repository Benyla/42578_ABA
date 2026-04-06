# Data

Flat pool used by the project and by RF-DETR training preparation:

```txt
data/
├── README.md
├── images/           # all JPEGs / PNGs
└── instances.json    # COCO annotations (file_name = basename only)
```

**Training** uses RF-DETR’s Roboflow layout. A generated copy lives under `data/rf_train/` (`train/` and `valid/` with `_annotations.coco.json`). Do not commit `data/rf_train/` (gitignored).

Commands:

```bash
uv sync --extra train
uv run aba-train prepare-data
uv run aba-train train --device cuda
```

`prepare-data` copies `instances.json` to `_annotations.coco.json` and syncs images. Validation currently mirrors training until you split a validation set.

**Auto-orientation** (EXIF) was applied in your Roboflow export; if you add raw images elsewhere, open them in a tool that applies EXIF orientation before training, or preprocess with a small script.
