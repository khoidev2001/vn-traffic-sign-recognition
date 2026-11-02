"""Trích chọn đặc trưng cho KNN / SVM: HOG (hình dạng, nét) + histogram màu HSV.

Biển báo giao thông phân biệt nhau chủ yếu bằng hình dạng (tròn/tam giác/vuông/bát giác),
màu viền (đỏ/xanh/vàng) và nét bên trong, nên ghép HOG với histogram màu.
"""
import cv2
import numpy as np
from skimage.feature import hog

HSV_BINS = (8, 4, 4)


def extract_features(crops_bgr):
    """crops_bgr: iterable ảnh BGR kích thước CROP_SIZE x CROP_SIZE -> ma trận (N, D) float32."""
    feats = []
    for crop in crops_bgr:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hog_vec = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys")
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, HSV_BINS, [0, 180, 0, 256, 0, 256]).ravel()
        hist = hist / (hist.sum() + 1e-6)
        feats.append(np.concatenate([hog_vec, hist]))
    return np.asarray(feats, dtype=np.float32)
