"""Ba bộ phân loại ảnh biển báo đã cắt: KNN, SVM (trên HOG + màu HSV) và CNN.

Cả ba cùng giao diện:  predict_proba(list_ảnh_BGR_64x64) -> mảng (N, số_lớp).
"""
import cv2
import joblib
import numpy as np
import torch
from torch import nn

from . import config
from .classes import num_classes
from .features import extract_features


class SignCNN(nn.Module):
    """CNN nhỏ: 4 khối Conv-BN-ReLU-MaxPool -> Global Average Pooling -> FC."""

    def __init__(self, n_classes):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def bgr_to_tensor(crops_bgr, device):
    """Danh sách ảnh BGR uint8 -> tensor (N,3,H,W) float trong [-1, 1] (RGB)."""
    arr = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops_bgr]).astype(np.float32)
    t = torch.from_numpy(arr).to(device).permute(0, 3, 1, 2)
    return t / 127.5 - 1.0


class SklearnClassifier:
    """KNN hoặc SVM đã huấn luyện (Pipeline: chuẩn hoá -> PCA -> bộ phân loại)."""

    def __init__(self, path):
        self.pipeline = joblib.load(path)

    def predict_proba(self, crops_bgr):
        proba = self.pipeline.predict_proba(extract_features(crops_bgr))
        full = np.zeros((len(crops_bgr), num_classes()), dtype=np.float32)
        full[:, self.pipeline.classes_] = proba
        return full


class CNNClassifier:
    def __init__(self, path=config.CNN_MODEL):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SignCNN(num_classes()).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, crops_bgr, batch_size=256):
        out = []
        for i in range(0, len(crops_bgr), batch_size):
            logits = self.model(bgr_to_tensor(crops_bgr[i:i + batch_size], self.device))
            out.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, num_classes()), dtype=np.float32)


CLASSIFIER_FILES = {"knn": config.KNN_MODEL, "svm": config.SVM_MODEL, "cnn": config.CNN_MODEL}


def load_classifier(name):
    if name == "cnn":
        return CNNClassifier()
    return SklearnClassifier(CLASSIFIER_FILES[name])
