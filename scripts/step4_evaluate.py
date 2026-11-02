"""Bước 4 - Đánh giá trên tập TEST (chưa từng dùng để huấn luyện hay chọn tham số).

Bộ phân loại (KNN/SVM/CNN) được chấm trên ảnh biển báo cắt theo khung gán nhãn chuẩn:
  Accuracy, Precision/Recall/F1 (macro + weighted), Top-5, thời gian mỗi biển, ma trận nhầm lẫn,
  báo cáo từng lớp, các cặp lớp hay nhầm nhất.
YOLOv8 được chấm bằng mAP, Precision, Recall trên ảnh gốc của tập test.

Chạy:  python scripts/step4_evaluate.py            (cả hai phần)
       python scripts/step4_evaluate.py --skip-detector
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from signrec import config  # noqa: E402
from signrec.classes import load_classes, num_classes  # noqa: E402
from signrec.classifiers import CLASSIFIER_FILES, load_classifier  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_confusion(cm, title, path):
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Lớp dự đoán")
    ax.set_ylabel("Lớp thực")
    ticks = np.arange(0, cm.shape[0], 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Tỉ lệ theo hàng")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def top_confusions(cm, names, top=15):
    pairs = [(int(cm[i, j]), i, j) for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i, j] > 0]
    pairs.sort(reverse=True)
    return [{"true_id": i, "true_name": names[i], "pred_id": j, "pred_name": names[j], "count": n}
            for n, i, j in pairs[:top]]


NOVEL_SIMILARITY = 0.97  # mẫu test có độ tương đồng với ảnh train gần nhất <= ngưỡng này được coi là "mẫu lạ"


def nearest_train_similarity(x_train, x_test):
    """Độ tương đồng cosine (ảnh xám 16x16, đã trừ trung bình) giữa mỗi ảnh test và ảnh train gần nhất."""
    def vec(images):
        v = np.stack([cv2.resize(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), (16, 16), interpolation=cv2.INTER_AREA).ravel()
                      for i in images]).astype(np.float32)
        v -= v.mean(axis=1, keepdims=True)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-6)

    return (vec(x_test) @ vec(x_train).T).max(axis=1)


def evaluate_classifiers():
    data = np.load(config.CROPS_FILE)
    x_test, y_test = data["x_test"], data["y_test"].astype(int)
    names = [c["name_vi"] for c in load_classes()]
    labels = sorted(set(y_test.tolist()))
    novel = nearest_train_similarity(data["x_train"], x_test) <= NOVEL_SIMILARITY
    out = {"n_test": int(len(y_test)), "n_classes_in_test": len(labels), "n_novel": int(novel.sum()),
           "novel_similarity_threshold": NOVEL_SIMILARITY, "models": {}, "top_confusions": {}}

    for name, path in CLASSIFIER_FILES.items():
        if not path.exists():
            print(f"[{name}] chưa huấn luyện, bỏ qua")
            continue
        clf = load_classifier(name)
        clf.predict_proba(x_test[:8])  # khởi động (nạp GPU / bộ nhớ đệm) trước khi bấm giờ
        t0 = time.perf_counter()
        proba = clf.predict_proba(list(x_test))
        ms_per_crop = (time.perf_counter() - t0) * 1000 / len(x_test)
        pred = proba.argmax(1)

        p, r, f1, _ = precision_recall_fscore_support(y_test, pred, labels=labels, average="macro", zero_division=0)
        _, _, f1_w, _ = precision_recall_fscore_support(y_test, pred, labels=labels, average="weighted", zero_division=0)
        top5 = float(np.mean([y in np.argsort(-pr)[:5] for y, pr in zip(y_test, proba)]))
        out["models"][name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f1), "f1_weighted": float(f1_w),
            "top5_accuracy": top5, "ms_per_crop": ms_per_crop,
            "accuracy_novel": float(accuracy_score(y_test[novel], pred[novel])),
        }
        print(f"[{name}] acc={out['models'][name]['accuracy']:.4f} F1={f1:.4f} top5={top5:.4f} {ms_per_crop:.2f} ms/biển")

        cm = confusion_matrix(y_test, pred, labels=range(num_classes()))
        pd.DataFrame(cm).to_csv(config.REPORTS_DIR / f"confusion_matrix_{name}.csv", index_label="true\\pred")
        plot_confusion(cm, f"Ma trận nhầm lẫn – {name.upper()} (tập test)", config.REPORTS_DIR / f"confusion_matrix_{name}.png")
        pp, rr, ff, ss = precision_recall_fscore_support(y_test, pred, labels=range(num_classes()), zero_division=0)
        pd.DataFrame({"id": range(num_classes()), "name_vi": names, "precision": pp, "recall": rr, "f1": ff, "support": ss}) \
            .to_csv(config.REPORTS_DIR / f"per_class_{name}.csv", index=False, encoding="utf-8-sig")
        out["top_confusions"][name] = top_confusions(cm, names)

    (config.REPORTS_DIR / "classifier_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_detector(imgsz=640):
    if not config.DETECTOR_MODEL.exists():
        print("[yolo] chưa có models/detector.pt, bỏ qua")
        return
    from ultralytics import YOLO

    yaml_path = config.ROOT / "data" / "detector.yaml"
    run_dir = config.MODELS_DIR / "detector_runs"
    model = YOLO(str(config.DETECTOR_MODEL))
    m = model.val(data=str(yaml_path), split="test", imgsz=imgsz, project=str(run_dir), name="test_eval",
                  exist_ok=True, plots=True, verbose=False)
    n_images = len(list((config.RAW_YOLO_DIR / "test" / "images").glob("*.jpg")))
    result = {
        "model": "YOLOv8n", "imgsz": imgsz, "n_images": n_images,
        "precision": float(m.box.mp), "recall": float(m.box.mr), "map50": float(m.box.map50), "map50_95": float(m.box.map),
        "speed_ms": {k: float(v) for k, v in m.speed.items()},
    }
    (config.REPORTS_DIR / "detector_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[yolo] P={result['precision']:.4f} R={result['recall']:.4f} mAP50={result['map50']:.4f} mAP50-95={result['map50_95']:.4f}")

    for src, dst in ((run_dir / "test_eval" / "confusion_matrix_normalized.png", "detector_confusion_matrix.png"),
                     (run_dir / "test_eval" / "BoxPR_curve.png", "detector_PR_curve.png"),
                     (run_dir / "train" / "results.png", "detector_results.png")):
        if src.exists():
            shutil.copy(src, config.REPORTS_DIR / dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-detector", action="store_true")
    parser.add_argument("--skip-classifiers", action="store_true")
    args = parser.parse_args()
    config.REPORTS_DIR.mkdir(exist_ok=True)
    if not args.skip_classifiers:
        evaluate_classifiers()
    if not args.skip_detector:
        evaluate_detector()


if __name__ == "__main__":
    main()
