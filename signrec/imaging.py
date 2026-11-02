"""Hàm xử lý ảnh dùng chung giữa chuẩn bị dữ liệu và webapp (đảm bảo cắt giống hệt nhau)."""
import cv2


def square_crop(img, cx, cy, side, out_size):
    """Cắt hình vuông tâm (cx, cy), cạnh `side`; phần lồi ra ngoài ảnh được nhân bản viền."""
    half = side / 2.0
    x1, y1 = int(round(cx - half)), int(round(cy - half))
    x2, y2 = int(round(cx + half)), int(round(cy + half))
    pad_l, pad_t = max(0, -x1), max(0, -y1)
    pad_r, pad_b = max(0, x2 - img.shape[1]), max(0, y2 - img.shape[0])
    if pad_l or pad_t or pad_r or pad_b:
        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
        x1, x2, y1, y2 = x1 + pad_l, x2 + pad_l, y1 + pad_t, y2 + pad_t
    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
