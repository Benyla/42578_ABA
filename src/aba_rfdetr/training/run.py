"""Run RF-DETR training from configs/training/dataset.yaml (Windows + CUDA friendly)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from rfdetr.config import TrainConfig

from aba_rfdetr.prepare_roboflow_layout import prepare_roboflow_layout


def _import_model_class(name: str):
    import rfdetr

    if not hasattr(rfdetr, name):
        raise ValueError(f"Unknown rfdetr model class {name!r}")
    return getattr(rfdetr, name)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _project_root_from_config(config_path: Path) -> Path:
    return config_path.resolve().parent.parent.parent


def _abs_str(p: str | Path, root: Path) -> str:
    path = Path(p)
    if not path.is_absolute():
        path = root / path
    return str(path.resolve())


def run_training(
    config_path: Path,
    *,
    device: str | None = None,
    dry_run: bool = False,
) -> None:
    """Load YAML, optionally prepare data layout, run ``model.train()``."""
    import typer

    raw = _load_yaml(config_path)
    root = _project_root_from_config(config_path)
    os.chdir(root)

    train_block = raw.get("training") or {}
    repeat_train = int(raw.get("repeat_train", 1))
    train_device = device or raw.get("device") or "cuda"
    model_cls_name = str((raw.get("model") or {}).get("class", "RFDETRMedium"))

    tc_keys = set(TrainConfig.model_fields.keys())
    train_kwargs: dict[str, Any] = {k: v for k, v in train_block.items() if k in tc_keys}

    if "dataset_dir" in train_kwargs:
        train_kwargs["dataset_dir"] = _abs_str(train_kwargs["dataset_dir"], root)
    if "output_dir" in train_kwargs:
        train_kwargs["output_dir"] = _abs_str(train_kwargs["output_dir"], root)

    if dry_run:
        TrainConfig(**train_kwargs)
        typer.echo(
            f"Dry run OK: model={model_cls_name}, device={train_device}, "
            f"repeat_train={repeat_train}, TrainConfig validates."
        )
        return

    prep = raw.get("prepare") or {}
    if prep.get("enabled", True):
        raw_data = Path(prep.get("raw_data_root", "data"))
        layout = Path(prep.get("roboflow_layout_dir", "data/rf_train"))
        if not raw_data.is_absolute():
            raw_data = root / raw_data
        if not layout.is_absolute():
            layout = root / layout
        prepare_roboflow_layout(raw_data, layout, force=bool(prep.get("force", False)))

    model_cfg = raw.get("model") or {}
    model_cls = _import_model_class(model_cls_name)
    model_kwargs = {k: v for k, v in model_cfg.items() if k != "class"}
    model = model_cls(**model_kwargs)

    if repeat_train > 1:
        import rfdetr.training.module_data as module_data

        original_dm = module_data.RFDETRDataModule
        r = repeat_train

        class RepeatedDM(original_dm):
            def setup(self, stage: str) -> None:
                super().setup(stage)
                if stage == "fit" and r > 1:
                    import torch.utils.data as tud

                    self._dataset_train = tud.ConcatDataset([self._dataset_train] * r)

            @property
            def class_names(self):
                import torch.utils.data as tud

                ds = self._dataset_train
                if isinstance(ds, tud.ConcatDataset) and len(ds.datasets) > 0:
                    ds = ds.datasets[0]
                coco = getattr(ds, "coco", None)
                if coco is not None and hasattr(coco, "cats"):
                    return [coco.cats[k]["name"] for k in sorted(coco.cats.keys())]
                return None

        module_data.RFDETRDataModule = RepeatedDM
        try:
            model.train(device=train_device, **train_kwargs)
        finally:
            module_data.RFDETRDataModule = original_dm
    else:
        model.train(device=train_device, **train_kwargs)
