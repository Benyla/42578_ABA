"""Standalone training entry point (Windows-safe).

On Windows, ``python -m aba_rfdetr.train`` causes multiprocessing DataLoader
workers to re-import the module via ``runpy.run_module``, triggering heavy
torch/torchvision init in every worker.  Running a plain script avoids that
because workers re-execute the *file* under a ``__name__ != "__main__"`` guard,
so only the lightweight imports at module level run.

Usage
-----
    uv run python scripts/train.py [--device cuda] [--dry-run]
    uv run python scripts/train.py --device cpu
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train RF-DETR (Windows-safe wrapper)")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "training" / "dataset.yaml",
        help="Path to training YAML config.",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda",
        help="Lightning accelerator device (cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and exit without training.",
    )
    args = parser.parse_args()

    from aba_rfdetr.training.run import run_training

    run_training(args.config, device=args.device, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
