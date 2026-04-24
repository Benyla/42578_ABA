from typing import Any

from pydantic import BaseModel, Field


class DetectionItem(BaseModel):
    """Single object detection result."""

    label: str
    class_id: int
    score: float = Field(ge=0.0, le=1.0)
    box_xyxy: list[float] = Field(
        description="Bounding box in pixel coordinates: xmin, ymin, xmax, ymax."
    )


class PredictResponse(BaseModel):
    success: bool
    detections: list[DetectionItem]
    error: str | None = None
    detail: dict[str, Any] | None = None


class CropResult(BaseModel):
    """One target crop with its Stage 2 detections."""

    crop_index: int
    crop_box_xyxy: list[float] = Field(
        description="Target crop region in original image coordinates."
    )
    crop_image_base64: str = Field(
        description="JPEG-encoded crop image as base64 string."
    )
    detections: list[DetectionItem] = Field(
        description="Stage 2 detections in crop-local coordinates."
    )
    target_box_local_xyxy: list[float] | None = Field(
        default=None,
        description=(
            "Stage-1 target box expressed in crop-local coordinates. "
            "Anything in the crop outside this box is padding."
        ),
    )
    crop_padding: float | None = Field(
        default=None,
        description="Fractional padding used when computing the crop box.",
    )
    predicted_type: int | None = Field(
        default=None, description="ResNet-predicted target type (1 or 2)."
    )
    prob_type2: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Probability of type 2."
    )
    crop_score: int = Field(
        default=0,
        description="Bullet score for this crop: sum of N for each Bullet_N detection.",
    )


class StagedPredictResponse(BaseModel):
    success: bool
    greyscale_image_base64: str | None = Field(
        default=None,
        description="Greyscale-converted full image as base64 JPEG (shown as preprocessing step).",
    )
    stage1_detections: list[DetectionItem] = Field(
        description="Target detections from Stage 1 on full image."
    )
    crops: list[CropResult] = Field(default_factory=list)
    total_score: int | None = Field(
        default=None,
        description="Sum of crop_score across all crops (bullet-based final score).",
    )
    error: str | None = None
    detail: dict[str, Any] | None = None


class TypePredictResponse(BaseModel):
    """Target type classification (type 1 vs type 2)."""

    success: bool
    predicted_type: int | None = Field(default=None, description="Predicted target type (1 or 2).")
    prob_type2: float | None = Field(default=None, ge=0.0, le=1.0, description="Probability of type 2.")
    crop_box_xyxy: list[float] | None = Field(
        default=None, description="Crop region used for classification (original image coords)."
    )
    error: str | None = None
    detail: dict[str, Any] | None = None
