"""Load RF-DETR and run inference; map supervision.Detections to API JSON."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import yaml
from PIL import Image

from aba_rfdetr.coco_labels import COCO_CLASS_NAMES
from aba_rfdetr.schemas import DetectionItem

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "inference.yaml"

_MODEL_LOCK = Lock()
_model: Any = None
_config_cache: dict[str, Any] | None = None
_class_names: list[str] | None = None


def project_root() -> Path:
    return _PROJECT_ROOT


def _config_path() -> Path:
    raw = os.environ.get("ABA_CONFIG_PATH", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_CONFIG


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def load_inference_config() -> dict[str, Any]:
    global _config_cache
    if _config_cache is None:
        _config_cache = _load_yaml(_config_path())
    return _config_cache


def _resolve_path_maybe(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    p = Path(value)
    if not p.is_absolute():
        p = (_PROJECT_ROOT / p).resolve()
    return str(p)


def _load_class_names(cfg: dict[str, Any]) -> list[str]:
    path = cfg.get("class_names_file")
    if path:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = (_PROJECT_ROOT / resolved).resolve()
        with resolved.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
            raise ValueError("class_names_file must be a JSON array of strings")
        return list(data)
    return list(COCO_CLASS_NAMES)


def get_class_names() -> list[str]:
    global _class_names
    if _class_names is None:
        cfg = load_inference_config()
        _class_names = _load_class_names(cfg)
    return _class_names


def _get_model_cls(name: str):
    import rfdetr

    registry = {
        "RFDETRNano": rfdetr.RFDETRNano,
        "RFDETRSmall": rfdetr.RFDETRSmall,
        "RFDETRMedium": rfdetr.RFDETRMedium,
        "RFDETRBase": rfdetr.RFDETRBase,
        "RFDETRLarge": rfdetr.RFDETRLarge,
    }
    if name not in registry:
        raise ValueError(f"Unknown model_class {name!r}. Choose one of: {sorted(registry)}")
    return registry[name]


def get_or_create_model():
    """Lazy-load RF-DETR once (thread-safe)."""
    global _model
    with _MODEL_LOCK:
        if _model is not None:
            return _model
        cfg = load_inference_config()
        cls_name = str(cfg.get("model_class", "RFDETRBase"))
        model_cls = _get_model_cls(cls_name)
        num_classes = int(cfg.get("num_classes", 80))
        ck = os.environ.get("MODEL_PATH", "").strip() or cfg.get("checkpoint_path")
        ck_resolved = _resolve_path_maybe(str(ck) if ck else None)
        kwargs: dict[str, Any] = {"num_classes": num_classes}
        if ck_resolved:
            kwargs["pretrain_weights"] = ck_resolved
        _model = model_cls(**kwargs)
        return _model


def reset_model_cache_for_tests() -> None:
    """Clear cached model and config (used from tests)."""
    global _model, _config_cache, _class_names
    with _MODEL_LOCK:
        _model = None
        _config_cache = None
        _class_names = None


def detections_to_items(raw, class_names: list[str]) -> list[DetectionItem]:
    """Convert supervision.Detections to DetectionItem list."""
    items: list[DetectionItem] = []
    if raw is None or len(raw) == 0:
        return items
    xyxy = raw.xyxy
    conf = raw.confidence
    cid = raw.class_id
    for i in range(len(raw)):
        c = int(cid[i])
        label = class_names[c] if 0 <= c < len(class_names) else f"class_{c}"
        items.append(
            DetectionItem(
                label=label,
                class_id=c,
                score=float(conf[i]),
                box_xyxy=[float(x) for x in xyxy[i].tolist()],
            )
        )
    return items


def predict_pil_image(image: Image.Image) -> list[DetectionItem]:
    """Run RF-DETR on a PIL image (RGB)."""
    cfg = load_inference_config()
    threshold = float(cfg.get("score_threshold", 0.35))
    model = get_or_create_model()
    det = model.predict(image, threshold=threshold)
    names = get_class_names()
    return detections_to_items(det, names)


def predict_image_bytes(data: bytes) -> list[DetectionItem]:
    """Decode bytes to PIL and predict."""
    import io

    img = Image.open(io.BytesIO(data)).convert("RGB")
    return predict_pil_image(img)


def predict_numpy_rgb(arr: np.ndarray) -> list[DetectionItem]:
    """Predict from HxWx3 uint8 RGB numpy array."""
    img = Image.fromarray(arr)
    return predict_pil_image(img)
