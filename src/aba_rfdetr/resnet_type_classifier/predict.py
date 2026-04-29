from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import models, transforms

from aba_rfdetr import inference as det_inference


@dataclass(frozen=True)
class TypePrediction:
    """Binary prediction: type 1 vs type 2."""

    predicted_type: int  # 1 or 2
    prob_type2: float
    crop_box_xyxy: list[float] | None = None


_TYPE_MODEL: Any = None
_TYPE_TF: Any = None


def _load_type_model(checkpoint_path: str | Path) -> tuple[torch.nn.Module, Any, int]:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "resnet18")
    if arch != "resnet18":
        raise ValueError(f"Unsupported arch: {arch!r}")
    image_size = int(ckpt.get("image_size", 224))

    m = models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(ckpt["model"])
    m.eval()

    tf = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return m, tf, image_size


def get_or_create_type_model() -> tuple[torch.nn.Module, Any]:
    """Lazy-load classifier from env `TYPE_MODEL_PATH` or default runs path."""
    global _TYPE_MODEL, _TYPE_TF
    if _TYPE_MODEL is not None and _TYPE_TF is not None:
        return _TYPE_MODEL, _TYPE_TF

    import os

    ck = os.environ.get("TYPE_MODEL_PATH", "").strip() or "runs/resnet_type12/model_best.pt"
    _TYPE_MODEL, _TYPE_TF, _ = _load_type_model(ck)
    return _TYPE_MODEL, _TYPE_TF


@torch.no_grad()
def predict_type_from_pil_crop(crop: Image.Image, *, device: str | None = None) -> TypePrediction:
    model, tf = get_or_create_type_model()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)

    x = tf(crop).unsqueeze(0).to(dev)
    logits = model(x)
    prob = torch.softmax(logits, dim=1)[0, 1].item()  # P(type2)
    predicted_type = 2 if prob >= 0.5 else 1
    return TypePrediction(predicted_type=predicted_type, prob_type2=float(prob), crop_box_xyxy=None)


def _crop_target_from_stage1(image: Image.Image, *, padding: float = 0.05) -> tuple[Image.Image, list[float]] | None:
    """Use Stage-1 detector to crop a single target ROI (top-1 by confidence)."""
    grey = det_inference._to_greyscale_rgb(image)  # reuse exact conversion
    with det_inference._MODEL_LOCK:
        s1 = det_inference._get_stage1_model()

    s1_det = s1.predict(grey, threshold=0.0)
    if s1_det is None or len(s1_det) == 0:
        return None

    # pick highest-confidence detection
    conf = [float(x) for x in s1_det.confidence.tolist()]
    i = max(range(len(conf)), key=lambda k: conf[k])
    x1, y1, x2, y2 = [float(v) for v in s1_det.xyxy[i].tolist()]

    img_w, img_h = image.size
    cx1, cy1, cx2, cy2 = det_inference._padded_crop_box(x1, y1, x2, y2, img_w, img_h, padding)
    crop = det_inference._to_greyscale_rgb(image.crop((cx1, cy1, cx2, cy2)))
    return crop, [float(cx1), float(cy1), float(cx2), float(cy2)]


@torch.no_grad()
def predict_type_from_image_bytes(data: bytes, *, padding: float = 0.05, device: str | None = None) -> TypePrediction:
    """Full pipeline: decode -> stage1 crop -> greyscale -> resnet."""
    img = det_inference._open_image(data)
    out = _crop_target_from_stage1(img, padding=padding)
    if out is None:
        return TypePrediction(predicted_type=1, prob_type2=0.0, crop_box_xyxy=None)
    crop, box = out
    pred = predict_type_from_pil_crop(crop, device=device)
    return TypePrediction(predicted_type=pred.predicted_type, prob_type2=pred.prob_type2, crop_box_xyxy=box)

