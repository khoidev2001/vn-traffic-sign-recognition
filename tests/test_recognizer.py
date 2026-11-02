import numpy as np
import pytest

from signrec import config
from signrec import recognizer as recognizer_module
from signrec.recognizer import SignRecognizer


def point_models_at(monkeypatch, folder):
    monkeypatch.setattr(config, "DETECTOR_MODEL", folder / "detector.pt")
    monkeypatch.setattr(recognizer_module, "CLASSIFIER_FILES", {name: folder / f"{name}.bin" for name in ("cnn", "svm", "knn")})


def test_nothing_is_available_without_trained_models(monkeypatch, tmp_path):
    point_models_at(monkeypatch, tmp_path)
    assert SignRecognizer().available() == {"yolo": False, "cnn": False, "svm": False, "knn": False}


def test_everything_is_available_when_model_files_exist(monkeypatch, tmp_path):
    point_models_at(monkeypatch, tmp_path)
    for name in ("detector.pt", "cnn.bin", "svm.bin", "knn.bin"):
        (tmp_path / name).write_bytes(b"x")
    assert all(SignRecognizer().available().values())


class FixedProbabilities:
    """Bộ phân loại giả: lớp 7 (0.7), lớp 3 (0.2), lớp 9 (0.1)."""

    def predict_proba(self, crops):
        proba = np.zeros((len(crops), 82), np.float32)
        proba[:, 7], proba[:, 3], proba[:, 9] = 0.7, 0.2, 0.1
        return proba


def test_single_sign_mode_classifies_the_whole_image(monkeypatch):
    rec = SignRecognizer()
    monkeypatch.setattr(rec, "_get_classifier", lambda name: FixedProbabilities())
    detections, _ms = rec.predict_single(np.zeros((120, 80, 3), np.uint8), model="yolo")

    (best,) = detections
    assert best["class_id"] == 7 and best["whole_image"] is True
    assert best["box"] == [0.0, 0.0, 1.0, 1.0]
    assert best["model_used"] == "cnn"                      # YOLO không có bước phân loại riêng nên dùng CNN
    assert [round(a["conf"], 2) for a in best["alternatives"]] == [0.2, 0.1]


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        SignRecognizer().predict(np.zeros((32, 32, 3), np.uint8), model="resnet")
