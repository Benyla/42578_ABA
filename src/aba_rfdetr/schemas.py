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
