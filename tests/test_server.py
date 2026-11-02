import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.server as server

NO_MODELS = {"yolo": False, "cnn": False, "svm": False, "knn": False}
ALL_MODELS = {"yolo": True, "cnn": True, "svm": True, "knn": True}


@pytest.fixture()
def client():
    return TestClient(server.app)


def jpeg_bytes():
    ok, buf = cv2.imencode(".jpg", np.zeros((64, 96, 3), np.uint8))
    assert ok
    return buf.tobytes()


def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200 and "Nhận dạng biển báo" in response.text
    assert response.headers["cache-control"] == "no-cache"


def test_classes_endpoint_lists_82_classes(client):
    data = client.get("/api/classes").json()
    assert len(data["classes"]) == 82 and "cam" in data["groups"]


def test_report_endpoint_always_returns_all_sections(client):
    assert set(client.get("/api/report").json()) == {"dataset", "training", "classifiers", "detector", "images"}


def test_predict_rejects_empty_and_non_image_bodies(client):
    assert client.post("/api/predict", content=b"").status_code == 400
    assert client.post("/api/predict", content=b"not an image").status_code == 400


def test_predict_returns_503_when_the_model_is_not_trained(client, monkeypatch):
    monkeypatch.setattr(server.recognizer, "available", lambda: NO_MODELS)
    assert client.get("/api/models").json()["available"] == NO_MODELS
    assert client.post("/api/predict?model=yolo", content=jpeg_bytes()).status_code == 503


def test_predict_clamps_parameters_before_calling_the_recognizer(client, monkeypatch):
    seen = {}

    def fake_predict(frame, model, conf, imgsz):
        seen.update(model=model, conf=conf, imgsz=imgsz, shape=frame.shape)
        return [], 1.0

    monkeypatch.setattr(server.recognizer, "available", lambda: ALL_MODELS)
    monkeypatch.setattr(server.recognizer, "predict", fake_predict)
    response = client.post("/api/predict?model=cnn&conf=5&imgsz=9999", content=jpeg_bytes())

    assert response.status_code == 200
    assert response.json() == {"width": 96, "height": 64, "ms": 1.0, "detections": []}
    assert seen == {"model": "cnn", "conf": 0.95, "imgsz": 1280, "shape": (64, 96, 3)}


def test_single_sign_flag_uses_whole_image_classification(client, monkeypatch):
    monkeypatch.setattr(server.recognizer, "available", lambda: ALL_MODELS)
    monkeypatch.setattr(server.recognizer, "predict_single", lambda frame, model: ([{"class_id": 0, "whole_image": True}], 2.0))
    response = client.post("/api/predict?model=cnn&single=true", content=jpeg_bytes())
    assert response.status_code == 200 and response.json()["detections"][0]["whole_image"] is True
