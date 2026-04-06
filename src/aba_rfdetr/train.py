"""CLI: prepare Roboflow layout + train RF-DETR."""

from __future__ import annotations

from pathlib import Path

import typer

from aba_rfdetr.prepare_roboflow_layout import prepare_roboflow_layout
from aba_rfdetr.training.run import run_training

app = typer.Typer(help="Prepare data layout and train RF-DETR.")


@app.command("prepare-data")
def prepare_data_cmd(
    raw_root: Path = Path("data"),
    layout_root: Path = Path("data/rf_train"),
    force: bool = typer.Option(False, "--force", "-f", help="Rebuild even if layout exists."),
    val_fraction: float = typer.Option(0.2, "--val-fraction", help="Fraction of images for validation."),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducible split."),
) -> None:
    """Split data/images + instances.json into data/rf_train/train|valid for rfdetr."""
    prepare_roboflow_layout(
        raw_root.resolve(), layout_root.resolve(),
        force=force, val_fraction=val_fraction, seed=seed,
    )


@app.command("train")
def train_cmd(
    config: Path = Path("configs/training/dataset.yaml"),
    device: str = typer.Option(
        "cuda",
        "--device",
        "-d",
        help="Lightning device (cuda, cuda:0, cpu).",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config and exit without training."),
) -> None:
    """Train RF-DETR using configs/training/dataset.yaml (runs prepare-data first unless disabled in YAML)."""
    run_training(config, device=device, dry_run=dry_run)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
