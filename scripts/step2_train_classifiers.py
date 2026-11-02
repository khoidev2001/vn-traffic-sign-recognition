"""Bước 2 - Huấn luyện các bộ phân loại trên ảnh biển báo đã cắt (data/crops.npz).

  KNN : HOG + HSV -> chuẩn hoá -> PCA -> K-Nearest Neighbors (chọn k theo tập val)
  SVM : cùng đặc trưng, SVM nhân RBF (chọn C theo tập val)
  CNN : mạng tích chập nhỏ, huấn luyện trên GPU với tăng cường dữ liệu

Tập test KHÔNG được dùng ở đây; việc chấm điểm nằm ở step4_evaluate.py.

Chạy:  python scripts/step2_train_classifiers.py            (cả ba)
       python scripts/step2_train_classifiers.py --only cnn
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from signrec import config  # noqa: E402
from signrec.classes import num_classes  # noqa: E402
from signrec.classifiers import SignCNN, bgr_to_tensor  # noqa: E402
from signrec.features import extract_features  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PCA_COMPONENTS = 150
SEED = 42


def load_crops():
    d = np.load(config.CROPS_FILE)
    return {k: d[k] for k in d.files}


def train_sklearn(name, data):
    """Huấn luyện KNN hoặc SVM; trả về thống kê tham số + độ chính xác trên tập val."""
    t0 = time.time()
    x_train = extract_features(data["x_train"])
    x_val = extract_features(data["x_val"])
    y_train, y_val = data["y_train"], data["y_val"]
    print(f"[{name}] đặc trưng: {x_train.shape[1]} chiều, trích xong sau {time.time() - t0:.0f}s")

    scaler = StandardScaler().fit(x_train)
    pca = PCA(PCA_COMPONENTS, random_state=SEED).fit(scaler.transform(x_train))
    z_train = pca.transform(scaler.transform(x_train))
    z_val = pca.transform(scaler.transform(x_val))
    print(f"[{name}] PCA {PCA_COMPONENTS} thành phần giữ {pca.explained_variance_ratio_.sum():.1%} phương sai")

    grid = {}
    if name == "knn":
        for k in (1, 3, 5, 7, 9, 11):
            clf = KNeighborsClassifier(k, weights="distance", n_jobs=-1).fit(z_train, y_train)
            grid[k] = float((clf.predict(z_val) == y_val).mean())
            print(f"[knn] k={k}: val acc = {grid[k]:.4f}")
        best = max(grid, key=grid.get)
        final = KNeighborsClassifier(best, weights="distance", n_jobs=-1).fit(z_train, y_train)
        params = {"k": best}
    else:
        for c in (1, 10, 100):
            clf = SVC(C=c, kernel="rbf", gamma="scale").fit(z_train, y_train)
            grid[c] = float((clf.predict(z_val) == y_val).mean())
            print(f"[svm] C={c}: val acc = {grid[c]:.4f}  ({time.time() - t0:.0f}s)")
        best = max(grid, key=grid.get)
        # probability=True để webapp có độ tin cậy (tốn thêm cross-validation nội bộ)
        final = SVC(C=best, kernel="rbf", gamma="scale", probability=True, random_state=SEED).fit(z_train, y_train)
        params = {"C": best}

    pipeline = Pipeline([("scale", scaler), ("pca", pca), ("clf", final)])
    path = config.KNN_MODEL if name == "knn" else config.SVM_MODEL
    joblib.dump(pipeline, path, compress=3)
    val_acc = float((pipeline.predict(x_val) == y_val).mean())
    print(f"[{name}] val acc cuối = {val_acc:.4f}, đã lưu {path.name} ({time.time() - t0:.0f}s)")
    return {"params": params, "grid_val_acc": {str(k): v for k, v in grid.items()}, "val_acc": val_acc,
            "pca_components": PCA_COMPONENTS, "train_seconds": round(time.time() - t0, 1)}


def augment(x, strength=1.0):
    """Tăng cường dữ liệu trên GPU, mỗi ảnh một tham số ngẫu nhiên. x: (N,3,H,W) trong [-1,1].

    KHÔNG lật ngang (sẽ đổi nghĩa biển rẽ trái/phải). Có mô phỏng ảnh mờ/độ phân giải thấp
    vì biển báo xa trong khung hình camera thường chỉ vài chục pixel.
    """
    n = x.shape[0]
    dev = x.device
    ang = (torch.rand(n, device=dev) * 2 - 1) * math.radians(12) * strength
    scale = 1 + (torch.rand(n, device=dev) * 2 - 1) * 0.15 * strength
    shift = (torch.rand(n, 2, device=dev) * 2 - 1) * 0.10 * strength
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], 1), torch.stack([sin, cos, shift[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)

    # giảm độ phân giải ngẫu nhiên cho một nửa số ảnh
    low = torch.rand(n, device=dev) < 0.5
    if low.any():
        size = int(torch.randint(20, 56, (1,)).item())
        small = F.interpolate(x[low], size=(size, size), mode="area")
        x[low] = F.interpolate(small, size=x.shape[-2:], mode="bilinear", align_corners=False)

    contrast = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * 0.35 * strength
    bright = (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * 0.30 * strength
    x = x * contrast + bright
    x = x + torch.randn_like(x) * 0.04 * strength
    return x.clamp(-1, 1)


@torch.no_grad()
def evaluate_cnn(model, x, y, batch=512):
    model.eval()
    correct, loss_sum = 0, 0.0
    for i in range(0, len(x), batch):
        logits = model(x[i:i + batch])
        loss_sum += F.cross_entropy(logits, y[i:i + batch], reduction="sum").item()
        correct += (logits.argmax(1) == y[i:i + batch]).sum().item()
    return loss_sum / len(x), correct / len(x)


def train_cnn(data, epochs=45, batch_size=128):
    torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = bgr_to_tensor(data["x_train"], dev)
    x_val = bgr_to_tensor(data["x_val"], dev)
    y_train = torch.from_numpy(data["y_train"].astype(np.int64)).to(dev)
    y_val = torch.from_numpy(data["y_val"].astype(np.int64)).to(dev)

    model = SignCNN(num_classes()).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    steps = epochs * math.ceil(len(x_train) / batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    print(f"[cnn] thiết bị={dev}, {n_params / 1e6:.2f}M tham số, {epochs} epoch")

    history, best_acc, t0 = [], 0.0, time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(x_train), device=dev)
        run_loss, run_correct = 0.0, 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = augment(x_train[idx]), y_train[idx]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item() * len(idx)
            run_correct += (logits.argmax(1) == yb).sum().item()
        val_loss, val_acc = evaluate_cnn(model, x_val, y_val)
        rec = {"epoch": epoch, "train_loss": run_loss / len(x_train), "train_acc": run_correct / len(x_train),
               "val_loss": val_loss, "val_acc": val_acc}
        history.append(rec)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config.CNN_MODEL)
        print(f"[cnn] epoch {epoch:2d}/{epochs} loss {rec['train_loss']:.3f} acc {rec['train_acc']:.3f} | "
              f"val loss {val_loss:.3f} acc {val_acc:.4f}  ({time.time() - t0:.0f}s)")

    plot_history(history)
    return {"val_acc": best_acc, "epochs": epochs, "params": n_params, "batch_size": batch_size,
            "train_seconds": round(time.time() - t0, 1), "history": history}


def plot_history(history):
    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, [h["train_loss"] for h in history], label="train")
    a1.plot(ep, [h["val_loss"] for h in history], label="val")
    a1.set_title("Loss (CNN)")
    a1.set_xlabel("Epoch")
    a1.legend()
    a2.plot(ep, [h["train_acc"] for h in history], label="train")
    a2.plot(ep, [h["val_acc"] for h in history], label="val")
    a2.set_title("Accuracy (CNN)")
    a2.set_xlabel("Epoch")
    a2.legend()
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "cnn_training_curves.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("knn", "svm", "cnn"))
    parser.add_argument("--epochs", type=int, default=45)
    args = parser.parse_args()

    config.MODELS_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    data = load_crops()
    summary_path = config.REPORTS_DIR / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    for name in ("knn", "svm", "cnn"):
        if args.only and args.only != name:
            continue
        summary[name] = train_cnn(data, args.epochs) if name == "cnn" else train_sklearn(name, data)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
