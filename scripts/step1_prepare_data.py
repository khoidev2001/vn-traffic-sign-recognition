"""Bước 1 - Chuẩn bị dữ liệu.

Đọc bộ ảnh gán nhãn YOLO (train/val/test) và cắt từng biển báo ra thành ảnh vuông
64x64 để huấn luyện/đánh giá các bộ phân loại (KNN, SVM, CNN).

Đầu ra:
  data/crops.npz                      mảng ảnh + nhãn cho từng tập
  assets/samples/<id>.jpg             ảnh đại diện của mỗi lớp (cho webapp)
  reports/dataset_stats.json          thống kê số mẫu
  reports/class_distribution.png      biểu đồ phân bố lớp
  reports/sample_grid.png             lưới ảnh mẫu (dùng chèn vào báo cáo)

Chạy:  python scripts/step1_prepare_data.py
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from signrec import config  # noqa: E402
from signrec.classes import load_classes  # noqa: E402
from signrec.imaging import square_crop  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SPLITS = ("train", "val", "test")


def read_split(split, best_by_class=None):
    images_dir = config.RAW_YOLO_DIR / split / "images"
    labels_dir = config.RAW_YOLO_DIR / split / "labels"
    xs, ys, names, sizes = [], [], [], []
    skipped_small = 0
    malformed = []
    for label_file in sorted(labels_dir.glob("*.txt")):
        lines = [ln.split() for ln in label_file.read_text().splitlines() if ln.strip()]
        # dòng nhãn đúng chuẩn YOLO phải có 5 cột; dòng lỗi (vd. train/8380.txt) bị bỏ qua
        malformed += [label_file.name for ln in lines if len(ln) != 5]
        lines = [ln for ln in lines if len(ln) == 5]
        if not lines:
            continue
        img = cv2.imread(str(images_dir / (label_file.stem + ".jpg")))
        if img is None:
            continue
        h, w = img.shape[:2]
        for cls, cx, cy, bw, bh in lines:
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            if min(bw, bh) < config.MIN_BOX_PIXELS:
                skipped_small += 1
                continue
            side = max(bw, bh) * config.CROP_MARGIN
            xs.append(square_crop(img, cx, cy, side, config.CROP_SIZE))
            sizes.append(max(bw, bh))
            # ảnh đại diện mỗi lớp: lấy biển to nhất trong tập train, cắt lại từ ảnh gốc ở 128px
            if best_by_class is not None:
                known = best_by_class.get(int(cls))
                if known is None or max(bw, bh) > known[1]:
                    best_by_class[int(cls)] = (square_crop(img, cx, cy, side, 128), max(bw, bh))
            ys.append(int(cls))
            names.append(label_file.stem)
    return np.stack(xs), np.array(ys, dtype=np.int16), np.array(names), np.array(sizes, dtype=np.float32), skipped_small, malformed


def save_sample_images(train_x_full):
    """Mỗi lớp lấy mẫu có kích thước gốc lớn nhất làm ảnh đại diện."""
    config.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for cls, (crop, _) in train_x_full.items():
        cv2.imwrite(str(config.SAMPLES_DIR / f"{cls}.jpg"), crop)


def plot_distribution(counts, classes):
    fig, ax = plt.subplots(figsize=(15, 5))
    width = 0.28
    ids = np.arange(len(classes))
    for k, split in enumerate(SPLITS):
        ax.bar(ids + (k - 1) * width, counts[split], width, label=split)
    ax.set_xlabel("ID lớp")
    ax.set_ylabel("Số mẫu")
    ax.set_title("Phân bố số mẫu theo lớp")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)


def plot_sample_grid(x, y, num_classes, per_row=14):
    rng = np.random.default_rng(0)
    rows = int(np.ceil(num_classes / per_row))
    fig, axes = plt.subplots(rows, per_row, figsize=(per_row * 1.1, rows * 1.2))
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i < num_classes:
            idx = np.flatnonzero(y == i)
            if len(idx):
                ax.imshow(cv2.cvtColor(x[rng.choice(idx)], cv2.COLOR_BGR2RGB))
            ax.set_title(str(i), fontsize=7, pad=1)
    fig.tight_layout(pad=0.3)
    fig.savefig(config.REPORTS_DIR / "sample_grid.png", dpi=150)
    plt.close(fig)


def main():
    config.REPORTS_DIR.mkdir(exist_ok=True)
    classes = load_classes()
    n_cls = len(classes)
    data, stats, counts = {}, {"crop_size": config.CROP_SIZE, "margin": config.CROP_MARGIN, "splits": {}}, {}
    best_by_class = {}  # cls -> (ảnh 128px, cạnh gốc)

    for split in SPLITS:
        x, y, names, sizes, skipped, malformed = read_split(split, best_by_class if split == "train" else None)
        data[f"x_{split}"], data[f"y_{split}"], data[f"id_{split}"] = x, y, names
        counts[split] = np.bincount(y, minlength=n_cls)
        stats["splits"][split] = {
            "images": int(len(set(names.tolist()))),
            "samples": int(len(y)),
            "skipped_small": int(skipped),
            "malformed_label_lines": malformed,
            "min_samples_per_class": int(counts[split][counts[split] > 0].min()),
            "max_samples_per_class": int(counts[split].max()),
            "classes_present": int((counts[split] > 0).sum()),
        }
        print(f"{split}: {len(y)} mẫu, bỏ {skipped} khung nhỏ")

    np.savez_compressed(config.CROPS_FILE, **data)
    stats["per_class"] = {str(c["id"]): {"name_vi": c["name_vi"], **{s: int(counts[s][c["id"]]) for s in SPLITS}} for c in classes}
    (config.REPORTS_DIR / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    save_sample_images(best_by_class)
    plot_distribution(counts, classes)
    plot_sample_grid(data["x_train"], data["y_train"], n_cls)
    print("Xong. Lớp thiếu ảnh mẫu:", [i for i in range(n_cls) if i not in best_by_class])


if __name__ == "__main__":
    main()
