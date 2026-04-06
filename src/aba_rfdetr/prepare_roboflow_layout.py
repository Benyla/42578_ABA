"""Build RF-DETR–compatible Roboflow layout from flat data/images + data/instances.json."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import typer


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_roboflow_layout(
    raw_data_root: Path,
    layout_root: Path,
    *,
    force: bool = False,
) -> None:
    """Create layout_root/{train,valid}/ with _annotations.coco.json and images.

    Validation mirrors training (no separate split yet). Images in ``valid`` are
    hardlinked to ``train`` when possible to save disk space (fallback: copy).
    """
    images_src = raw_data_root / "images"
    ann_src = raw_data_root / "instances.json"
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

    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(ann_src, train_dir / "_annotations.coco.json")
    shutil.copy2(ann_src, valid_dir / "_annotations.coco.json")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    count = 0
    for f in sorted(images_src.iterdir()):
        if f.suffix.lower() not in exts:
            continue
        t_train = train_dir / f.name
        shutil.copy2(f, t_train)
        _link_or_copy(t_train, valid_dir / f.name)
        count += 1

    if count == 0:
        raise RuntimeError(f"No image files found under {images_src}")

    typer.echo(f"Prepared {count} images under {train_dir} and {valid_dir}.")
