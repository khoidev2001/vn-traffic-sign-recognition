"""Webapp nhận dạng biển báo giao thông Việt Nam (FastAPI).

Chạy:  python -m uvicorn app.server:app --port 8000      rồi mở http://localhost:8000
(Camera trình duyệt chỉ hoạt động trên localhost hoặc https.)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from signrec import config  # noqa: E402
from signrec.classes import GROUP_LABELS, load_classes  # noqa: E402
from signrec.recognizer import SignRecognizer  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

app = FastAPI(title="Nhận dạng biển báo giao thông Việt Nam")
recognizer = SignRecognizer()


@app.middleware("http")
async def no_cache_for_frontend(request: Request, call_next):
    """Khi đang sửa giao diện, luôn tải bản mới nhất thay vì bản trong bộ nhớ đệm."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-cache"
    return response


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@app.get("/api/models")
def models():
    return {"available": recognizer.available(), "device": "GPU (CUDA)" if recognizer.device != "cpu" else "CPU"}


@app.get("/api/classes")
def classes():
    stats = (read_json(config.REPORTS_DIR / "dataset_stats.json") or {}).get("per_class", {})
    return {
        "groups": GROUP_LABELS,
        "classes": [{**c, "train_samples": stats.get(str(c["id"]), {}).get("train")} for c in load_classes()],
    }


@app.get("/api/report")
def report():
    """Số liệu để trang 'Đánh giá' hiển thị; file nào chưa có thì trả null."""
    r = config.REPORTS_DIR
    return {
        "dataset": read_json(r / "dataset_stats.json"),
        "training": read_json(r / "training_summary.json"),
        "classifiers": read_json(r / "classifier_metrics.json"),
        "detector": read_json(r / "detector_metrics.json"),
        "images": sorted(p.name for p in r.glob("*.png")),
    }


@app.post("/api/predict")
async def predict(request: Request, model: str = "yolo", conf: float = 0.4, imgsz: int = 640, single: bool = False):
    """Body: byte của ảnh JPEG/PNG. Trả về các biển báo tìm thấy (toạ độ chuẩn hoá 0..1)."""
    body = await request.body()
    if not body or len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "Ảnh rỗng hoặc quá lớn (tối đa 15 MB)")
    frame = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Không đọc được ảnh")
    if not recognizer.available().get(model):
        raise HTTPException(503, f"Mô hình '{model}' chưa được huấn luyện")
    if single:
        detections, ms = recognizer.predict_single(frame, model=model)
        return {"width": frame.shape[1], "height": frame.shape[0], "ms": round(ms, 1), "detections": detections}
    detections, ms = recognizer.predict(
        frame, model=model, conf=min(max(conf, 0.05), 0.95), imgsz=min(max(imgsz // 32 * 32, 320), 1280)
    )
    return {"width": frame.shape[1], "height": frame.shape[0], "ms": round(ms, 1), "detections": detections}


# Bản tải mới từ GitHub chưa có các thư mục này (chúng được sinh ra khi chạy step1/step4)
config.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/samples", StaticFiles(directory=config.SAMPLES_DIR), name="samples")
app.mount("/reports", StaticFiles(directory=config.REPORTS_DIR), name="reports")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
