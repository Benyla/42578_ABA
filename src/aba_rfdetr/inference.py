"""Two-stage cascade inference: Stage 1 (Target) -> crop -> Stage 2 (Bullets)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageOps

from aba_rfdetr.schemas import CropResult, DetectionItem, StagedPredictResponse

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "inference.yaml"

_MODEL_LOCK = Lock()
_stage1_model: Any = None
_stage2_model: Any = None
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


def _resolve_path(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    p = Path(value)
    if not p.is_absolute():
        p = (_PROJECT_ROOT / p).resolve()
    return str(p)


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


def _build_model(stage_cfg: dict[str, Any]) -> Any:
    cls_name = str(stage_cfg.get("model_class", "RFDETRMedium"))
    model_cls = _get_model_cls(cls_name)
    num_classes = int(stage_cfg.get("num_classes", 1))
    ck = stage_cfg.get("checkpoint_path")
    ck_resolved = _resolve_path(str(ck) if ck else None)
    kwargs: dict[str, Any] = {"num_classes": num_classes}
    if ck_resolved:
        kwargs["pretrain_weights"] = ck_resolved
    return model_cls(**kwargs)


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
    return [
        "Target",
        "Bullet_0", "Bullet_1", "Bullet_10", "Bullet_2", "Bullet_3",
        "Bullet_4", "Bullet_5", "Bullet_6", "Bullet_7", "Bullet_8",
        "Bullet_9", "black_contour",
    ]


def get_class_names() -> list[str]:
    global _class_names
    if _class_names is None:
        cfg = load_inference_config()
        _class_names = _load_class_names(cfg)
    return _class_names


def _get_stage1_model() -> Any:
    global _stage1_model
    if _stage1_model is not None:
        return _stage1_model
    cfg = load_inference_config()
    s1 = cfg.get("stage1", {})
    ck_env = os.environ.get("STAGE1_MODEL_PATH", "").strip()
    if ck_env:
        s1 = dict(s1, checkpoint_path=ck_env)
    _stage1_model = _build_model(s1)
    return _stage1_model


def _get_stage2_model() -> Any:
    global _stage2_model
    if _stage2_model is not None:
        return _stage2_model
    cfg = load_inference_config()
    s2 = cfg.get("stage2", {})
    ck_env = os.environ.get("STAGE2_MODEL_PATH", "").strip()
    if ck_env:
        s2 = dict(s2, checkpoint_path=ck_env)
    _stage2_model = _build_model(s2)
    return _stage2_model


def get_or_create_model():
    """Lazy-load both models (thread-safe). Returns (stage1, stage2)."""
    with _MODEL_LOCK:
        return _get_stage1_model(), _get_stage2_model()


def reset_model_cache_for_tests() -> None:
    """Clear cached models and config (used from tests)."""
    global _stage1_model, _stage2_model, _config_cache, _class_names
    with _MODEL_LOCK:
        _stage1_model = None
        _stage2_model = None
        _config_cache = None
        _class_names = None


# 12-class names for the dedicated Stage 2 model (matches NEW_CATEGORIES
# in prepare_stage2_crops.py).
_STAGE2_CLASS_NAMES_12 = [
    "Bullet_0", "Bullet_1", "Bullet_10", "Bullet_2", "Bullet_3",
    "Bullet_4", "Bullet_5", "Bullet_6", "Bullet_7", "Bullet_8",
    "Bullet_9", "black_contour",
]

# 14-class names for the temporary all-in-one model used before a dedicated
# Stage 2 model is trained.  Matches the original COCO category id order
# (id 0 = Shooting-Discs placeholder, ids 1-13 = actual classes).
_STAGE2_CLASS_NAMES_14 = [
    "Shooting-Discs",
    "Bullet_0", "Bullet_1", "Bullet_10", "Bullet_2", "Bullet_3",
    "Bullet_4", "Bullet_5", "Bullet_6", "Bullet_7", "Bullet_8",
    "Bullet_9", "Target", "black_contour",
]


def _stage2_class_names() -> list[str]:
    cfg = load_inference_config()
    n = int(cfg.get("stage2", {}).get("num_classes", 12))
    if n == 14:
        return _STAGE2_CLASS_NAMES_14
    return _STAGE2_CLASS_NAMES_12


def _padded_crop_box(
    x1: float, y1: float, x2: float, y2: float,
    img_w: int, img_h: int, padding: float,
) -> tuple[int, int, int, int]:
    """Return (cx1, cy1, cx2, cy2) for the padded crop, clamped to image."""
    w = x2 - x1
    h = y2 - y1
    pad_x = w * padding
    pad_y = h * padding
    return (
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(img_w, int(x2 + pad_x)),
        min(img_h, int(y2 + pad_y)),
    )


def predict_pil_image(image: Image.Image) -> list[DetectionItem]:
    """Run two-stage cascade on a PIL image (RGB).

    1. Stage 1 detects Target bboxes on the full image.
    2. For each Target, crop the *original* full-res image to the bbox
       (with configurable padding), run Stage 2 to detect bullets.
    3. Remap Stage 2 detections back to original image coordinates.
    4. Return Target + bullet detections combined.
    """
    cfg = load_inference_config()
    s1_cfg = cfg.get("stage1", {})
    s2_cfg = cfg.get("stage2", {})
    s1_threshold = float(s1_cfg.get("score_threshold", 0.35))
    s2_threshold = float(s2_cfg.get("score_threshold", 0.35))
    crop_padding = float(s2_cfg.get("crop_padding", 0.10))
    names = get_class_names()

    with _MODEL_LOCK:
        s1_model = _get_stage1_model()
        s2_model = _get_stage2_model()

    # --- Stage 1: detect Targets on full image ---
    s1_det = s1_model.predict(image, threshold=s1_threshold)
    items: list[DetectionItem] = []

    if s1_det is None or len(s1_det) == 0:
        return items

    img_w, img_h = image.size

    for i in range(len(s1_det)):
        x1, y1, x2, y2 = s1_det.xyxy[i].tolist()
        score = float(s1_det.confidence[i])
        items.append(DetectionItem(
            label="Target",
            class_id=0,
            score=score,
            box_xyxy=[x1, y1, x2, y2],
        ))

        # --- Stage 2: detect bullets on crop ---
        cx1, cy1, cx2, cy2 = _padded_crop_box(x1, y1, x2, y2, img_w, img_h, crop_padding)
        crop = image.crop((cx1, cy1, cx2, cy2))
        s2_det = s2_model.predict(crop, threshold=s2_threshold)

        if s2_det is None or len(s2_det) == 0:
            continue

        for j in range(len(s2_det)):
            bx1, by1, bx2, by2 = s2_det.xyxy[j].tolist()
            cid = int(s2_det.class_id[j])
            s2_names = _stage2_class_names()
            label = s2_names[cid] if 0 <= cid < len(s2_names) else f"class_{cid}"
            # class_id in combined output: Target=0, then stage2 classes offset by 1
            combined_cid = cid + 1
            items.append(DetectionItem(
                label=label,
                class_id=combined_cid,
                score=float(s2_det.confidence[j]),
                box_xyxy=[bx1 + cx1, by1 + cy1, bx2 + cx1, by2 + cy1],
            ))

    return items


def predict_pil_image_staged(image: Image.Image) -> StagedPredictResponse:
    """Run two-stage cascade and return structured intermediate results.

    Unlike ``predict_pil_image`` this returns the crop images (as base64 JPEG)
    and Stage 2 detections in **crop-local** coordinates so the frontend can
    visualise each pipeline step.
    """
    import base64
    import io

    cfg = load_inference_config()
    s1_cfg = cfg.get("stage1", {})
    s2_cfg = cfg.get("stage2", {})
    s1_threshold = float(s1_cfg.get("score_threshold", 0.35))
    s2_threshold = float(s2_cfg.get("score_threshold", 0.35))
    crop_padding = float(s2_cfg.get("crop_padding", 0.10))
    s2_names = _stage2_class_names()

    with _MODEL_LOCK:
        s1_model = _get_stage1_model()
        s2_model = _get_stage2_model()

    s1_det = s1_model.predict(image, threshold=s1_threshold)
    stage1_items: list[DetectionItem] = []
    crops: list[CropResult] = []

    if s1_det is None or len(s1_det) == 0:
        return StagedPredictResponse(
            success=True, stage1_detections=[], crops=[]
        )

    img_w, img_h = image.size

    for i in range(len(s1_det)):
        x1, y1, x2, y2 = s1_det.xyxy[i].tolist()
        score = float(s1_det.confidence[i])
        stage1_items.append(DetectionItem(
            label="Target",
            class_id=0,
            score=score,
            box_xyxy=[x1, y1, x2, y2],
        ))

        cx1, cy1, cx2, cy2 = _padded_crop_box(
            x1, y1, x2, y2, img_w, img_h, crop_padding
        )
        crop = image.crop((cx1, cy1, cx2, cy2))

        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=90)
        crop_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        s2_det = s2_model.predict(crop, threshold=s2_threshold)
        crop_dets: list[DetectionItem] = []
        if s2_det is not None and len(s2_det) > 0:
            for j in range(len(s2_det)):
                bx1, by1, bx2, by2 = s2_det.xyxy[j].tolist()
                cid = int(s2_det.class_id[j])
                label = s2_names[cid] if 0 <= cid < len(s2_names) else f"class_{cid}"
                crop_dets.append(DetectionItem(
                    label=label,
                    class_id=cid,
                    score=float(s2_det.confidence[j]),
                    box_xyxy=[bx1, by1, bx2, by2],
                ))

        crops.append(CropResult(
            crop_index=i,
            crop_box_xyxy=[float(cx1), float(cy1), float(cx2), float(cy2)],
            crop_image_base64=crop_b64,
            detections=crop_dets,
        ))

    return StagedPredictResponse(
        success=True,
        stage1_detections=stage1_items,
        crops=crops,
    )


def _open_image(data: bytes) -> Image.Image:
    """Decode bytes, apply EXIF rotation, and convert to RGB."""
    import io

    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def predict_image_bytes(data: bytes) -> list[DetectionItem]:
    """Decode bytes to PIL and predict."""
    return predict_pil_image(_open_image(data))


def predict_image_bytes_staged(data: bytes) -> StagedPredictResponse:
    """Decode bytes to PIL and run staged prediction."""
    return predict_pil_image_staged(_open_image(data))


def predict_numpy_rgb(arr: np.ndarray) -> list[DetectionItem]:
    """Predict from HxWx3 uint8 RGB numpy array."""
    img = Image.fromarray(arr)
    return predict_pil_image(img)
