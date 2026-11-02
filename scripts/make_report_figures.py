"""Sinh các hình dùng trong báo cáo Word (ghi vào reports/fig_*.png).

Cần chạy sau step1-step4. Chạy lại khi có số liệu/mô hình mới để hình luôn khớp.

Chạy:  python scripts/make_report_figures.py
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signrec import config  # noqa: E402
from signrec.classes import load_classes  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = "DejaVu Sans"   # có đủ dấu tiếng Việt
OUT = config.REPORTS_DIR
BLUE, GREEN, ORANGE, GREY = "#2f6fed", "#12a594", "#e89b0c", "#6b7686"
MODEL_COLORS = {"knn": ORANGE, "svm": GREEN, "cnn": BLUE}


def box(ax, x, y, w, h, text, color, fontsize=9, text_color="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=text_color, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=GREY, lw=1.4))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    box(ax, 0.1, 1.4, 1.5, 0.8, "Camera /\nVideo / Ảnh", GREY)
    box(ax, 2.0, 1.4, 1.6, 0.8, "Thu nhỏ khung hình\n(≤ 960 px) + JPEG", GREY)
    box(ax, 4.0, 1.4, 1.7, 0.8, "YOLOv8n\nphát hiện khung\nbiển báo", BLUE)
    box(ax, 6.1, 1.4, 1.6, 0.8, "Cắt vuông\n64×64, chuẩn hoá", ORANGE)
    box(ax, 8.1, 2.45, 1.4, 0.65, "CNN", GREEN)
    box(ax, 8.1, 1.5, 1.4, 0.65, "SVM\n(HOG+PCA)", GREEN)
    box(ax, 8.1, 0.55, 1.4, 0.65, "KNN\n(HOG+PCA)", GREEN)
    box(ax, 9.8, 1.4, 1.1, 0.8, "Khung +\ntên + độ\ntin cậy", BLUE, fontsize=8.5)
    for x1, x2 in ((1.6, 2.0), (3.6, 4.0), (5.7, 6.1)):
        arrow(ax, x1, 1.8, x2, 1.8)
    for y in (2.77, 1.82, 0.87):
        arrow(ax, 7.7, 1.8, 8.1, y)
        arrow(ax, 9.5, y, 9.8, 1.8)
    ax.text(5.5, 0.25, "Chế độ \"YOLO một giai đoạn\": dùng luôn lớp do YOLO dự đoán, bỏ qua bước cắt + phân loại",
            ha="center", fontsize=8.5, color=GREY, style="italic")
    fig.savefig(OUT / "fig_pipeline.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_cnn_architecture():
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    stages = [
        ("Ảnh vào\n64×64×3", GREY, 0.1, 1.1),
        ("Khối 1\n2×Conv3×3(32)\nBN+ReLU, MaxPool\n→ 32×32×32", BLUE, 1.5, 1.7),
        ("Khối 2\n2×Conv3×3(64)\n→ 16×16×64", BLUE, 3.4, 1.5),
        ("Khối 3\n2×Conv3×3(128)\n→ 8×8×128", BLUE, 5.1, 1.5),
        ("Khối 4\n2×Conv3×3(256)\n→ 4×4×256", BLUE, 6.8, 1.5),
        ("GAP → 256\nDropout 0.3\nFC 256→82", GREEN, 8.5, 1.4),
        ("Softmax\n82 lớp", ORANGE, 10.05, 0.9),
    ]
    for text, color, x, w in stages:
        box(ax, x, 0.9, w, 1.4, text, color, fontsize=8.2)
    for i in range(len(stages) - 1):
        x_end = stages[i][2] + stages[i][3]
        arrow(ax, x_end, 1.6, stages[i + 1][2], 1.6)
    ax.text(5.5, 0.35, "Mỗi khối: Conv3×3 → BatchNorm → ReLU → Conv3×3 → BatchNorm → ReLU → MaxPool 2×2",
            ha="center", fontsize=9, color=GREY, style="italic")
    fig.savefig(OUT / "fig_cnn_architecture.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_metrics_comparison():
    data = json.loads((OUT / "classifier_metrics.json").read_text(encoding="utf-8"))["models"]
    keys = [("accuracy", "Accuracy"), ("precision_macro", "Precision"), ("recall_macro", "Recall"), ("f1_macro", "F1-Score"),
            ("accuracy_novel", "Accuracy\n(mẫu lạ)")]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    width = 0.26
    x = np.arange(len(keys))
    for i, name in enumerate(("knn", "svm", "cnn")):
        vals = [data[name][k] * 100 for k, _ in keys]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=name.upper(), color=MODEL_COLORS[name])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in keys])
    ax.set_ylim(90, 100.8)
    ax.set_ylabel("%")
    ax.set_title("So sánh KNN, SVM và CNN trên tập test")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_metrics_comparison.png", dpi=170)
    plt.close(fig)


def fig_lowest_f1():
    import pandas as pd

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)
    for ax, name in zip(axes, ("knn", "svm", "cnn")):
        df = pd.read_csv(OUT / f"per_class_{name}.csv", encoding="utf-8-sig")
        df = df[df["support"] > 0].sort_values("f1").head(8).iloc[::-1]
        labels = [f"#{r.id} {r.name_vi[:22]}" for r in df.itertuples()]
        ax.barh(labels, df["f1"] * 100, color=MODEL_COLORS[name])
        ax.set_title(f"{name.upper()}: 8 lớp F1 thấp nhất", fontsize=10)
        ax.set_xlim(40, 101)
        ax.grid(axis="x", alpha=0.3)
        ax.tick_params(axis="y", labelsize=7.5)
    axes[1].set_xlabel("F1-Score (%)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_lowest_f1.png", dpi=170)
    plt.close(fig)


def fig_preprocessing():
    from step2_train_classifiers import augment  # dùng đúng hàm tăng cường của lúc huấn luyện
    import torch

    from signrec.classifiers import bgr_to_tensor

    d = np.load(config.CROPS_FILE)
    x, y = d["x_train"], d["y_train"]
    rng = np.random.default_rng(3)
    classes = [0, 12, 24, 47, 64, 69]
    picks = [rng.choice(np.flatnonzero(y == c)) for c in classes]
    base = [x[i] for i in picks]
    torch.manual_seed(7)
    tensor = bgr_to_tensor(base, "cpu")
    variants = [augment(tensor.clone()) for _ in range(3)]
    fig, axes = plt.subplots(4, len(base), figsize=(9.5, 6.4))
    for j in range(len(base)):
        axes[0, j].imshow(cv2.cvtColor(base[j], cv2.COLOR_BGR2RGB))
        axes[0, j].set_title(f"lớp {classes[j]}", fontsize=8)
        for r in range(3):
            arr = ((variants[r][j].permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            axes[r + 1, j].imshow(arr)
    for r, name in enumerate(["Ảnh cắt 64×64", "Tăng cường #1", "Tăng cường #2", "Tăng cường #3"]):
        axes[r, 0].set_ylabel(name, fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "fig_preprocessing.png", dpi=170)
    plt.close(fig)


def fig_hog():
    from skimage.feature import hog

    d = np.load(config.CROPS_FILE)
    x, y = d["x_train"], d["y_train"]
    rng = np.random.default_rng(5)
    classes = [0, 12, 47]
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    for r, c in enumerate(classes):
        crop = x[rng.choice(np.flatnonzero(y == c))]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, hog_img = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [8], [0, 180]).ravel()
        axes[r, 0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axes[r, 1].imshow(hog_img, cmap="gray")
        axes[r, 2].bar(range(8), hist / hist.sum(), color=BLUE)
        axes[r, 2].set_ylim(0, 1)
        for c_ in (0, 1):
            axes[r, c_].axis("off")
    axes[0, 0].set_title("Ảnh 64×64", fontsize=9)
    axes[0, 1].set_title("Đặc trưng HOG", fontsize=9)
    axes[0, 2].set_title("Histogram kênh H (HSV)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_hog.png", dpi=170)
    plt.close(fig)


def fig_detection_examples():
    if not config.DETECTOR_MODEL.exists():
        return
    from ultralytics import YOLO

    classes = load_classes()
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 20)
    colors = {"cam": "#e5484d", "hieu_lenh": "#2f6fed", "nguy_hiem": "#e89b0c", "chi_dan": "#12a594", "den_tin_hieu": "#8e4ec6"}
    images = sorted((config.RAW_YOLO_DIR / "test" / "images").glob("*.jpg"))
    # ảnh có >=2 biển, chọn ngẫu nhiên có hạt giống cố định
    multi = [p for p in images if len((config.RAW_YOLO_DIR / "test" / "labels" / (p.stem + ".txt")).read_text().split("\n")) >= 3]
    picks = list(np.random.default_rng(11).choice(len(multi), 6, replace=False))
    model = YOLO(str(config.DETECTOR_MODEL))
    tiles = []
    for idx in picks:
        img = Image.open(multi[idx]).convert("RGB")
        res = model.predict(np.array(img)[:, :, ::-1], imgsz=640, conf=0.4, agnostic_nms=True, verbose=False)[0]
        draw = ImageDraw.Draw(img)
        for (x1, y1, x2, y2), cid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int)):
            col = colors[classes[cid]["group"]]
            draw.rectangle([x1, y1, x2, y2], outline=col, width=3)
            label = classes[cid]["name_vi"]
            tw = draw.textlength(label, font=font)
            lx = min(max(0, x1), max(0, img.width - tw - 8))   # không cho nhãn tràn ra ngoài ảnh
            ly = max(0, y1 - 26)
            draw.rectangle([lx, ly, lx + tw + 8, ly + 26], fill=col)
            draw.text((lx + 4, ly), label, font=font, fill="white")
        tiles.append(img.resize((640, 640)))
    rows = [np.hstack([np.array(t) for t in tiles[i:i + 3]]) for i in range(0, 6, 3)]
    Image.fromarray(np.vstack(rows)).save(OUT / "fig_detection_examples.jpg", quality=90)


def main():
    OUT.mkdir(exist_ok=True)
    for fn in (fig_pipeline, fig_cnn_architecture, fig_metrics_comparison, fig_lowest_f1, fig_preprocessing, fig_hog,
               fig_detection_examples):
        fn()
        print("ok", fn.__name__)


if __name__ == "__main__":
    main()
