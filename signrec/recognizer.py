"""Bộ nhận dạng dùng trong webapp: YOLO tìm biển báo, rồi chọn cách gán lớp.

Các chế độ (model):
  yolo : lấy luôn lớp do YOLO dự đoán
  cnn / svm / knn : lấy khung của YOLO, cắt biển báo rồi để bộ phân loại tương ứng quyết định lớp
"""
import threading
import time

import numpy as np
import torch

from . import config
from .classes import GROUP_LABELS, load_classes
from .classifiers import CLASSIFIER_FILES, load_classifier
from .imaging import square_crop

MODEL_NAMES = ("yolo", "cnn", "svm", "knn")


class SignRecognizer:
    def __init__(self):
        self._lock = threading.Lock()  # PyTorch/sklearn không an toàn khi gọi song song
        self._detector = None
        self._classifiers = {}
        self._classes = load_classes()
        self.device = 0 if torch.cuda.is_available() else "cpu"

    def available(self):
        have_detector = config.DETECTOR_MODEL.exists()
        return {
            "yolo": have_detector,
            **{n: have_detector and CLASSIFIER_FILES[n].exists() for n in ("cnn", "svm", "knn")},
        }

    def _get_detector(self):
        if self._detector is None:
            from ultralytics import YOLO

            self._detector = YOLO(str(config.DETECTOR_MODEL))
        return self._detector

    def _get_classifier(self, name):
        if name not in self._classifiers:
            self._classifiers[name] = load_classifier(name)
        return self._classifiers[name]

    def predict_single(self, frame_bgr, model="cnn", top_k=3):
        """Ảnh chỉ chứa MỘT biển báo (hình vẽ, ảnh cận cảnh): bỏ qua YOLO, phân loại cả ảnh.

        YOLO học từ ảnh chụp đường phố nên thường không tìm ra khung trên ảnh dạng này.
        """
        if model == "yolo":
            model = "cnn"  # YOLO không có bước phân loại riêng
        h, w = frame_bgr.shape[:2]
        t0 = time.perf_counter()
        crop = square_crop(frame_bgr, w / 2, h / 2, max(w, h), config.CROP_SIZE)
        with self._lock:
            proba = self._get_classifier(model).predict_proba([crop])[0]
        order = np.argsort(-proba)[:top_k]
        best = self._detection([0.0, 0.0, 1.0, 1.0], int(order[0]), float(proba[order[0]]), float(proba[order[0]]))
        best["whole_image"] = True
        best["model_used"] = model
        best["alternatives"] = [{"name_vi": self._classes[int(i)]["name_vi"], "conf": float(proba[i])} for i in order[1:]]
        return [best], (time.perf_counter() - t0) * 1000

    def _detection(self, box, class_id, conf, det_conf):
        info = self._classes[class_id]
        return {
            "box": box,
            "class_id": class_id,
            "name_vi": info["name_vi"],
            "name_en": info["name_en"],
            "group": info["group"],
            "group_label": GROUP_LABELS[info["group"]],
            "conf": conf,
            "det_conf": det_conf,
        }

    def predict(self, frame_bgr, model="yolo", conf=0.4, imgsz=640):
        """Trả về (danh sách phát hiện, thời gian xử lý ms). Toạ độ khung chuẩn hoá về [0,1]."""
        if model not in MODEL_NAMES:
            raise ValueError(f"model không hợp lệ: {model}")
        h, w = frame_bgr.shape[:2]
        t0 = time.perf_counter()
        with self._lock:
            result = self._get_detector().predict(
                frame_bgr, imgsz=imgsz, conf=min(conf, 0.25) if model != "yolo" else conf,
                agnostic_nms=True, device=self.device, half=self.device != "cpu", verbose=False,
            )[0]
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            det_conf = boxes.conf.cpu().numpy()
            yolo_cls = boxes.cls.cpu().numpy().astype(int)

            if model == "yolo" or len(xyxy) == 0:
                cls_ids, cls_conf = yolo_cls, det_conf
            else:
                crops = []
                for x1, y1, x2, y2 in xyxy:
                    side = max(x2 - x1, y2 - y1) * config.CROP_MARGIN
                    crops.append(square_crop(frame_bgr, (x1 + x2) / 2, (y1 + y2) / 2, side, config.CROP_SIZE))
                proba = self._get_classifier(model).predict_proba(crops)
                cls_ids, cls_conf = proba.argmax(1), proba.max(1)

        detections = []
        for (x1, y1, x2, y2), d_conf, cid, c_conf in zip(xyxy, det_conf, cls_ids, cls_conf):
            if model != "yolo" and (c_conf < conf or d_conf < 0.25):
                continue
            detections.append(self._detection(
                [float(x1 / w), float(y1 / h), float(x2 / w), float(y2 / h)], int(cid), float(c_conf), float(d_conf)))
        detections.sort(key=lambda d: -d["conf"])
        return detections, (time.perf_counter() - t0) * 1000
