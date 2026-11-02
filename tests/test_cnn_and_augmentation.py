import importlib.util

import numpy as np
import torch

from signrec import config
from signrec.classifiers import SignCNN, bgr_to_tensor


def load_training_script():
    spec = importlib.util.spec_from_file_location("step2_train_classifiers", config.ROOT / "scripts" / "step2_train_classifiers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cnn_forward_shape():
    model = SignCNN(82).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 64, 64))
    assert out.shape == (2, 82)


def test_bgr_to_tensor_scales_to_minus_one_one_and_swaps_to_rgb():
    blue_bgr = np.zeros((64, 64, 3), np.uint8)
    blue_bgr[..., 0] = 255                                  # kênh B của OpenCV
    t = bgr_to_tensor([blue_bgr], "cpu")
    assert t.shape == (1, 3, 64, 64)
    assert t[0, 2].min() == 1.0 and t[0, 0].max() == -1.0   # sang RGB: xanh dương nằm ở kênh cuối


def test_augmentation_keeps_shape_and_range():
    augment = load_training_script().augment
    torch.manual_seed(0)
    out = augment(torch.rand(16, 3, 64, 64) * 2 - 1)
    assert out.shape == (16, 3, 64, 64)
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_augmentation_never_flips_horizontally():
    """Lật ngang làm biển "cấm rẽ trái" thành "cấm rẽ phải" nên tăng cường dữ liệu tuyệt đối không được lật."""
    augment = load_training_script().augment
    torch.manual_seed(1)
    x = torch.full((64, 3, 64, 64), -1.0)
    x[..., :32] = 1.0                                       # nửa trái sáng, nửa phải tối
    out = augment(x)
    assert (out[..., :32].mean(dim=(1, 2, 3)) > out[..., 32:].mean(dim=(1, 2, 3))).all()
