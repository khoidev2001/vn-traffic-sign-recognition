"""Đo thời gian xử lý một khung hình của webapp cho từng phương pháp (YOLO, CNN, SVM, KNN).

Kết quả ghi vào reports/latency.json (dùng cho báo cáo). Chạy lại khi đã hoàn thiện để lấy số liệu cuối;
nên tắt các tác vụ nặng khác (huấn luyện, server) trước khi đo.

Chạy:  python scripts/benchmark_latency.py
"""
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from signrec import config  # noqa: E402
from signrec.recognizer import SignRecognizer  # noqa: E402

N_IMAGES = 20      # số ảnh test dùng để đo
N_WARMUP = 3


def main():
    images = sorted((config.RAW_YOLO_DIR / "test" / "images").glob("*.jpg"))[:N_IMAGES]
    frames = [cv2.imread(str(p)) for p in images]
    recognizer = SignRecognizer()
    result = {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
              "n_frames": len(frames), "imgsz": 640, "models": {}}
    for model in ("yolo", "cnn", "svm", "knn"):
        for f in frames[:N_WARMUP]:
            recognizer.predict(f, model=model)
        times = []
        for f in frames:
            t0 = time.perf_counter()
            recognizer.predict(f, model=model, imgsz=640)
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        result["models"][model] = {
            "mean_ms": statistics.mean(times), "median_ms": statistics.median(times),
            "p95_ms": times[int(0.95 * (len(times) - 1))], "fps": 1000 / statistics.mean(times),
        }
        print(f"{model}: {result['models'][model]['mean_ms']:.1f} ms/khung ({result['models'][model]['fps']:.1f} FPS)")
    (config.REPORTS_DIR / "latency.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
