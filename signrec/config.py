"""Đường dẫn và hằng số dùng chung cho cả script huấn luyện lẫn webapp."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_YOLO_DIR = ROOT / "data" / "yolo_raw" / "dataset"   # ảnh + nhãn YOLO gốc (train/val/test)
CROPS_FILE = ROOT / "data" / "crops.npz"                # ảnh biển báo đã cắt, 64x64
ASSETS_DIR = ROOT / "assets"
CLASSES_CSV = ASSETS_DIR / "classes.csv"
SAMPLES_DIR = ASSETS_DIR / "samples"                    # ảnh đại diện mỗi lớp (cho webapp)
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

KNN_MODEL = MODELS_DIR / "knn.joblib"
SVM_MODEL = MODELS_DIR / "svm.joblib"
CNN_MODEL = MODELS_DIR / "cnn.pt"
DETECTOR_MODEL = MODELS_DIR / "detector.pt"

CROP_SIZE = 64          # kích thước ảnh biển báo đưa vào bộ phân loại
CROP_MARGIN = 1.2       # cắt vuông rộng hơn khung biển báo 20%
MIN_BOX_PIXELS = 10     # bỏ khung nhỏ hơn 10 px (quá mờ để nhận dạng)

# Console Windows mặc định cp1252 sẽ lỗi khi in tiếng Việt
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
