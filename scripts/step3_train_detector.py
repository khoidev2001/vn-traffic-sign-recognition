"""Bước 3 - Huấn luyện bộ phát hiện + nhận dạng biển báo YOLOv8.

YOLO vừa định vị biển báo trong khung hình, vừa đoán lớp (82 lớp). Webapp dùng khung của YOLO,
rồi có thể chuyển phần nhận dạng sang CNN / SVM / KNN để so sánh.

Chạy:  python scripts/step3_train_detector.py --epochs 35
Kết quả: models/detector.pt (trọng số tốt nhất), models/detector_runs/ (log, biểu đồ huấn luyện)
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from signrec import config  # noqa: E402
from signrec.classes import load_classes  # noqa: E402


def write_dataset_yaml():
    path = config.ROOT / "data" / "detector.yaml"
    names = "\n".join(f"  {c['id']}: \"{c['name_en']}\"" for c in load_classes())
    path.write_text(
        f"path: {config.RAW_YOLO_DIR.as_posix()}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n{names}\n",
        encoding="utf-8",
    )
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--base", default="yolov8n.pt", help="trọng số khởi tạo (yolov8n.pt nhanh nhất, yolov8s.pt chính xác hơn)")
    args = parser.parse_args()

    from ultralytics import YOLO

    config.MODELS_DIR.mkdir(exist_ok=True)
    model = YOLO(str(config.MODELS_DIR / args.base))
    model.train(
        data=str(write_dataset_yaml()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=4,
        patience=10,
        project=str(config.MODELS_DIR / "detector_runs"),
        name="train",
        exist_ok=True,
        fliplr=0.0,  # KHÔNG lật ngang: lật sẽ biến "cấm rẽ trái" thành "cấm rẽ phải"
        plots=True,
    )
    best = config.MODELS_DIR / "detector_runs" / "train" / "weights" / "best.pt"
    shutil.copy(best, config.DETECTOR_MODEL)
    print("Đã lưu", config.DETECTOR_MODEL)


if __name__ == "__main__":
    main()
