# Nhận dạng biển báo giao thông Việt Nam (webapp)

[![CI](https://github.com/khoidev2001/vn-traffic-sign-recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/khoidev2001/vn-traffic-sign-recognition/actions/workflows/ci.yml)

Đồ án môn Lý thuyết nhận dạng. Hệ thống nhận dạng 82 lớp biển báo/đèn tín hiệu giao thông Việt Nam từ camera,
video hoặc ảnh, chạy trên trình duyệt (giao diện) + Python (xử lý).

## Kiến trúc

```
Camera/Video/Ảnh ──► trình duyệt (app/static) ──JPEG──► FastAPI (app/server.py)
                                                           │
                                       YOLOv8 tìm khung biển báo (models/detector.pt)
                                                           │
                         ┌────────── chọn cách gán lớp ────┴──────────┐
                     YOLO (một giai đoạn)        cắt 64×64 → CNN / SVM / KNN
                                                           │
                              khung + tên tiếng Việt + độ tin cậy ──► vẽ lên video
```

| Thành phần | Kỹ thuật |
|---|---|
| Phát hiện biển báo | YOLOv8n (Ultralytics), tắt lật ngang khi huấn luyện |
| CNN | 4 khối Conv-BN-ReLU-MaxPool + GAP + FC, ảnh 64×64, tăng cường dữ liệu trên GPU |
| SVM / KNN | HOG (9 hướng, cell 8×8) + histogram màu HSV → chuẩn hoá → PCA 150 chiều |

## Cấu trúc thư mục

```
signrec/          thư viện dùng chung (đường dẫn, danh sách lớp, đặc trưng, bộ phân loại, bộ nhận dạng)
scripts/
  step1_prepare_data.py        cắt biển báo từ nhãn YOLO → data/crops.npz, thống kê, ảnh mẫu
  step2_train_classifiers.py   huấn luyện KNN, SVM, CNN
  step3_train_detector.py      huấn luyện YOLOv8
  step4_evaluate.py            chấm điểm trên tập test, ma trận nhầm lẫn → reports/
app/              webapp (server.py + static/index.html, style.css, app.js)
assets/classes.csv            id, tên Anh, tên Việt, nhóm của 82 lớp (sửa trực tiếp được)
reports/          số liệu và hình dùng để viết báo cáo Word
```

## Cài đặt (Windows, GPU NVIDIA)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python -m pip install -r requirements.txt
```

## Dữ liệu

Bộ [Traffic-sign-detection-VietNam](https://huggingface.co/datasets/star092304/Traffic-sign-detection-VietNam)
(Hugging Face, **CC-BY-4.0**): ~10.000 ảnh, nhãn YOLO, chia sẵn train/val/test. Tải file
`traffic-sign-detection-vietnam.zip` (~900 MB), giải nén vào `data/yolo_raw/` sao cho có `data/yolo_raw/dataset/train/images/...`.

## Chạy

```powershell
.venv\Scripts\python scripts\step1_prepare_data.py
.venv\Scripts\python scripts\step2_train_classifiers.py
.venv\Scripts\python scripts\step3_train_detector.py --epochs 35
.venv\Scripts\python scripts\step4_evaluate.py

.venv\Scripts\python -m uvicorn app.server:app --port 8000
```

Mở <http://localhost:8000>. Camera trình duyệt chỉ cho dùng trên `localhost` hoặc `https`
(muốn thử bằng điện thoại cần chạy qua https).

## Hạn chế đã biết của dữ liệu

- **Lớp 57** ("Giới hạn tốc độ 10km/h") lẫn nhiều biển **40 km/h** → hay nhầm với lớp 2.
- Một số lớp trông giống nhau (vd. lớp 8 "Giảm tốc độ" và lớp 10 "Chướng ngại vật" cùng nền "ĐI CHẬM" ở một số ảnh).
- Khoảng 11% mẫu test có bản gần như trùng trong train → accuracy tổng lạc quan; bảng đánh giá có thêm cột
  "Acc (mẫu lạ)" để so sánh.
- File `train/labels/8380.txt` có một dòng thừa cột, bị bỏ qua khi cắt ảnh.
- Phần "nhóm" của lớp (cấm/hiệu lệnh/nguy hiểm/chỉ dẫn) trong `assets/classes.csv` do người làm đồ án gán thủ công,
  chỉ dùng để tô màu giao diện; cần đối chiếu QCVN 41:2019/BGTVT nếu đưa vào báo cáo.

## Kiểm thử và CI

```powershell
.venv\Scripts\python -m pip install -r requirements-ci.txt pytest
.venv\Scripts\python -m pytest -q          # kiểm thử nhanh, không cần dữ liệu hay mô hình
```

GitHub Actions (`.github/workflows/ci.yml`) chạy mỗi lần push lên `main` và mỗi pull request:

- **Lint:** `ruff check` (lỗi cú pháp, import thừa, tên chưa định nghĩa) và `node --check` cho `app/static/app.js`.
- **Kiểm thử:** `pytest` với PyTorch bản CPU, gồm: danh sách 82 lớp, cắt ảnh và đặc trưng HOG+HSV, hình dạng đầu ra của CNN,
  tăng cường dữ liệu **không lật ngang**, bộ nhận dạng (chế độ một biển báo, mô hình chưa huấn luyện), và các API của webapp.

CI **không** huấn luyện mô hình và không tải dữ liệu (nặng, cần GPU); phần đó chạy ở máy cá nhân theo các bước ở trên.

## Ghi công

- Bộ dữ liệu: [star092304/Traffic-sign-detection-VietNam](https://huggingface.co/datasets/star092304/Traffic-sign-detection-VietNam),
  giấy phép [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Các ảnh nhỏ trong `assets/samples/` được cắt từ bộ dữ liệu này.
- Phát hiện biển báo dùng [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0).

## Giấy phép

Mã nguồn của repo theo giấy phép [MIT](LICENSE). Lưu ý: repo chỉ chứa mã của đồ án, **không** chứa Ultralytics; thư viện này theo AGPL-3.0
nên nếu bạn phân phối hoặc triển khai dịch vụ dùng nó thì phải tuân theo điều khoản AGPL của họ. Dữ liệu và ảnh mẫu theo CC-BY-4.0 (xem trên).
