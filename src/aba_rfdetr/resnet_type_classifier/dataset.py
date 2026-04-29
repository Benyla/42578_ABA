from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int  # 0 or 1


class ImageFolderBinary(Dataset):
    """Minimal dataset for data/resnet_type12/{split}/{1,2}/*.jpg.

    Maps class '1' -> 0 and class '2' -> 1.
    """

    def __init__(self, root: Path, *, transform: Callable | None = None):
        self.root = Path(root)
        self.transform = transform
        self.samples: list[Sample] = []

        for class_name, label in (("1", 0), ("2", 1)):
            d = self.root / class_name
            if not d.exists():
                continue
            for p in sorted(d.rglob("*")):
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    self.samples.append(Sample(path=p, label=label))

        if not self.samples:
            raise ValueError(f"No images found under: {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, s.label

