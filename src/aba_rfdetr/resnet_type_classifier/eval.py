from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import models, transforms

from .dataset import ImageFolderBinary


def _build_model_from_checkpoint(ckpt: dict) -> torch.nn.Module:
    arch = ckpt.get("arch", "resnet18")
    if arch != "resnet18":
        raise ValueError(f"Unsupported arch in checkpoint: {arch!r}")
    m = models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(ckpt["model"])
    return m


@torch.no_grad()
def evaluate(model: torch.nn.Module, dl, device: str) -> dict:
    model.eval()
    cm = [[0, 0], [0, 0]]
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        for yi, pi in zip(y.tolist(), pred.tolist()):
            cm[yi][pi] += 1
    total = sum(sum(r) for r in cm)
    acc = (cm[0][0] + cm[1][1]) / max(1, total)
    tpr0 = cm[0][0] / max(1, (cm[0][0] + cm[0][1]))
    tpr1 = cm[1][1] / max(1, (cm[1][0] + cm[1][1]))
    bal_acc = 0.5 * (tpr0 + tpr1)
    return {"acc": acc, "balanced_acc": bal_acc, "cm": cm, "total": total}


def plot_confusion_matrix(cm: list[list[int]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion matrix (val)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1], labels=["type1", "type2"])
    ax.set_yticks([0, 1], labels=["type1", "type2"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate saved ResNet checkpoint on val set")
    ap.add_argument("--data-root", type=Path, default=Path("data/resnet_type12"))
    ap.add_argument("--checkpoint", type=Path, default=Path("runs/resnet_type12/model_best.pt"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs/resnet_type12"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    image_size = int(ckpt.get("image_size", 224))

    tf_val = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    ds_val = ImageFolderBinary(args.data_root / "val", transform=tf_val)
    dl_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = _build_model_from_checkpoint(ckpt).to(args.device)
    metrics = evaluate(model, dl_val, args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_confusion_matrix(metrics["cm"], args.out_dir / "confusion_matrix.png")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()

