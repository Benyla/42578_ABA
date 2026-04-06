"""Build RF-DETR–compatible Roboflow layout from flat data/images + data/instances.json."""

from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import typer


def _split_coco(
    coco: dict,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[dict, dict]:
    """Split a COCO dict into train/val by image, keeping annotations grouped."""
    image_ids = [img["id"] for img in coco["images"]]
    rng = random.Random(seed)
    rng.shuffle(image_ids)

    n_val = max(1, int(len(image_ids) * val_fraction))
    val_ids = set(image_ids[:n_val])
    train_ids = set(image_ids[n_val:])

    id_to_image = {img["id"]: img for img in coco["images"]}
    anns_by_image: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    shared = {k: coco[k] for k in ("info", "licenses", "categories") if k in coco}

    train_coco = {
        **shared,
        "images": [id_to_image[i] for i in sorted(train_ids)],
        "annotations": [a for i in sorted(train_ids) for a in anns_by_image.get(i, [])],
    }
    val_coco = {
        **shared,
        "images": [id_to_image[i] for i in sorted(val_ids)],
        "annotations": [a for i in sorted(val_ids) for a in anns_by_image.get(i, [])],
    }
    return train_coco, val_coco


def prepare_roboflow_layout(
    raw_data_root: Path,
    layout_root: Path,
    *,
    force: bool = False,
    val_fraction: float = 0.2,
    seed: int = 42,
    annotations_file: str = "instances.json",
    images_subdir: str = "images",
) -> None:
    """Create layout_root/{train,valid}/ with per-split annotations and images.

    Images are split by ``val_fraction`` (default 20 %) using a deterministic
    seed so results are reproducible.  Each split gets its own
    ``_annotations.coco.json`` containing only the relevant images/annotations.

    Parameters ``annotations_file`` and ``images_subdir`` allow reuse for
    stage-specific datasets (e.g. ``instances_stage1.json`` / ``crops``).
    """
    images_src = raw_data_root / images_subdir
    ann_src = raw_data_root / annotations_file
    if not ann_src.is_file():
        raise FileNotFoundError(f"Missing annotations: {ann_src}")
    if not images_src.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_src}")

    train_dir = layout_root / "train"
    valid_dir = layout_root / "valid"
    if layout_root.exists() and not force:
        if (train_dir / "_annotations.coco.json").is_file() and any(train_dir.glob("*.jpg")):
            typer.echo(f"Layout already present at {layout_root}; use --force to rebuild.")
            return

    coco = json.loads(ann_src.read_text(encoding="utf-8"))
    train_coco, val_coco = _split_coco(coco, val_fraction=val_fraction, seed=seed)

    if layout_root.exists():
        shutil.rmtree(layout_root)
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    train_dir.joinpath("_annotations.coco.json").write_text(
        json.dumps(train_coco), encoding="utf-8"
    )
    valid_dir.joinpath("_annotations.coco.json").write_text(
        json.dumps(val_coco), encoding="utf-8"
    )

    train_fnames = {img["file_name"] for img in train_coco["images"]}
    val_fnames = {img["file_name"] for img in val_coco["images"]}

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    t_count = v_count = 0
    for f in sorted(images_src.iterdir()):
        if f.suffix.lower() not in exts:
            continue
        if f.name in train_fnames:
            shutil.copy2(f, train_dir / f.name)
            t_count += 1
        elif f.name in val_fnames:
            shutil.copy2(f, valid_dir / f.name)
            v_count += 1

    if t_count == 0:
        raise RuntimeError(f"No training images found under {images_src}")

    typer.echo(
        f"Split {t_count + v_count} images: "
        f"{t_count} train, {v_count} valid ({val_fraction:.0%}) "
        f"under {layout_root}."
    )
