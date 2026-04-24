"""FastAPI app: health, JSON inference, and upload UI."""

from __future__ import annotations

import traceback
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from aba_rfdetr.inference import predict_image_bytes, predict_image_bytes_staged
from aba_rfdetr.schemas import PredictResponse, StagedPredictResponse, TypePredictResponse
from aba_rfdetr.resnet_type_classifier.predict import predict_type_from_image_bytes

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

app = FastAPI(title="ABA RF-DETR", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image file (image/*).")
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file.")
        items = predict_image_bytes(data)
        return PredictResponse(success=True, detections=items, error=None)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — surface message to client in controlled way
        return PredictResponse(
            success=False,
            detections=[],
            error=str(exc),
            detail={"traceback": traceback.format_exc()},
        )


@app.post("/predict/staged", response_model=StagedPredictResponse)
async def predict_staged(file: UploadFile = File(...)) -> StagedPredictResponse:
    """Two-stage prediction returning intermediate crops and per-crop detections."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image file (image/*).")
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file.")
        return predict_image_bytes_staged(data)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        return StagedPredictResponse(
            success=False,
            stage1_detections=[],
            crops=[],
            error=str(exc),
            detail={"traceback": traceback.format_exc()},
        )


@app.post("/predict/type", response_model=TypePredictResponse)
async def predict_type(file: UploadFile = File(...)) -> TypePredictResponse:
    """Classify target type (1 vs 2) using Stage-1 crop + ResNet."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image file (image/*).")
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file.")
        pred = predict_type_from_image_bytes(data)
        return TypePredictResponse(
            success=True,
            predicted_type=pred.predicted_type,
            prob_type2=pred.prob_type2,
            crop_box_xyxy=pred.crop_box_xyxy,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        return TypePredictResponse(
            success=False,
            predicted_type=None,
            prob_type2=None,
            crop_box_xyxy=None,
            error=str(exc),
            detail={"traceback": traceback.format_exc()},
        )


def create_app() -> FastAPI:
    """Factory for uvicorn string import."""
    return app
