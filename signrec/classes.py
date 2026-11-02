"""Danh sách 82 lớp biển báo: id, tên tiếng Anh, tên tiếng Việt, nhóm.

Sửa tên/nhóm trực tiếp trong assets/classes.csv, không cần đổi code.
"""
import csv
from functools import lru_cache

from .config import CLASSES_CSV

GROUP_LABELS = {
    "cam": "Biển cấm",
    "hieu_lenh": "Biển hiệu lệnh",
    "nguy_hiem": "Biển cảnh báo nguy hiểm",
    "chi_dan": "Biển chỉ dẫn",
    "den_tin_hieu": "Đèn tín hiệu",
}


@lru_cache(maxsize=1)
def load_classes():
    with open(CLASSES_CSV, encoding="utf-8-sig", newline="") as f:
        rows = [{**r, "id": int(r["id"])} for r in csv.DictReader(f)]
    rows.sort(key=lambda r: r["id"])
    return rows


def class_names_vi():
    return [r["name_vi"] for r in load_classes()]


def num_classes():
    return len(load_classes())
