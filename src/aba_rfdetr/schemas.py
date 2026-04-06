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


class StagedPredictResponse(BaseModel):
    success: bool
    stage1_detections: list[DetectionItem] = Field(
        description="Target detections from Stage 1 on full image."
    )
    crops: list[CropResult] = Field(default_factory=list)
    error: str | None = None
    detail: dict[str, Any] | None = None
