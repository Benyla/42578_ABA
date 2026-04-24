from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

from .dataset import ImageFolderBinary


@dataclass
class TrainConfig:
    data_root: Path
    out_dir: Path
    epochs: int = 12
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    seed: int = 42
    image_size: int = 224
    freeze_backbone_epochs: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_model() -> nn.Module:
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


@torch.no_grad()
def _eval(model: nn.Module, loader, device: str) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    # confusion matrix for 2 classes
    cm = [[0, 0], [0, 0]]
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        for yi, pi in zip(y.tolist(), pred.tolist()):
            cm[yi][pi] += 1
        correct += (pred == y).sum().item()
        total += y.numel()

    acc = correct / max(1, total)
    # balanced accuracy
    tpr0 = cm[0][0] / max(1, (cm[0][0] + cm[0][1]))
    tpr1 = cm[1][1] / max(1, (cm[1][0] + cm[1][1]))
    bal_acc = 0.5 * (tpr0 + tpr1)
    return {"acc": acc, "balanced_acc": bal_acc, "cm00": cm[0][0], "cm01": cm[0][1], "cm10": cm[1][0], "cm11": cm[1][1]}


def train(cfg: TrainConfig) -> Path:
    _set_seed(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    tf_train = transforms.Compose(
        [
            transforms.Resize(cfg.image_size + 32),
            transforms.CenterCrop(cfg.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    tf_val = transforms.Compose(
        [
            transforms.Resize(cfg.image_size + 32),
            transforms.CenterCrop(cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    ds_train = ImageFolderBinary(cfg.data_root / "train", transform=tf_train)
    ds_val = ImageFolderBinary(cfg.data_root / "val", transform=tf_val)

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        generator=g,
    )
    dl_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    device = cfg.device
    model = _build_model().to(device)

    # freeze backbone initially
    for name, p in model.named_parameters():
        if not name.startswith("fc."):
            p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_path = cfg.out_dir / "model_best.pt"
    best_bal = -1.0
    history: list[dict] = []

    for epoch in range(cfg.epochs):
        if epoch == cfg.freeze_backbone_epochs:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=cfg.lr * 0.3, weight_decay=cfg.weight_decay)

        model.train()
        t0 = time.time()
        running = 0.0
        seen = 0

        for x, y in dl_train:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * y.size(0)
            seen += y.size(0)

        train_loss = running / max(1, seen)
        metrics = _eval(model, dl_val, device)
        rec = {"epoch": epoch, "train_loss": train_loss, "seconds": time.time() - t0, **metrics}
        history.append(rec)
        print(json.dumps(rec))

        if metrics["balanced_acc"] > best_bal:
            best_bal = metrics["balanced_acc"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "arch": "resnet18",
                    "image_size": cfg.image_size,
                    "class_mapping": {"1": 0, "2": 1},
                    "metrics": rec,
                    "config": asdict(cfg),
                },
                best_path,
            )

    (cfg.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (cfg.out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")
    return best_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Train small ResNet on cached type1-vs-type2 crops")
    ap.add_argument("--data-root", type=Path, default=Path("data/resnet_type12"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs/resnet_type12"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--freeze-backbone-epochs", type=int, default=3)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    cfg = TrainConfig(
        data_root=args.data_root,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        image_size=args.image_size,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        device=args.device,
    )
    best = train(cfg)
    print(f"Best checkpoint: {best}")


if __name__ == "__main__":
    main()

