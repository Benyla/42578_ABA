from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from aba_rfdetr.api import app
from aba_rfdetr.schemas import DetectionItem


def _tiny_png() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (32, 32), color=(120, 80, 200)).save(buf, format="PNG")
    return buf.getvalue()


def test_health() -> None:
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_index() -> None:
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "RF-DETR" in r.text


def test_predict_rejects_non_image() -> None:
    client = TestClient(app)
    r = client.post(
        "/predict",
        files={"file": ("x.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 400


def test_predict_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_predict(data: bytes) -> list[DetectionItem]:
        assert data
        return [
            DetectionItem(
                label="cat",
                class_id=15,
                score=0.91,
                box_xyxy=[1.0, 2.0, 30.0, 40.0],
            )
        ]

    monkeypatch.setattr("aba_rfdetr.api.predict_image_bytes", fake_predict)
    client = TestClient(app)
    r = client.post(
        "/predict",
        files={"file": ("t.png", _tiny_png(), "image/png")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert len(body["detections"]) == 1
    assert body["detections"][0]["label"] == "cat"
    assert body["detections"][0]["score"] == pytest.approx(0.91)
