import numpy as np

from signrec import config
from signrec.features import extract_features
from signrec.imaging import square_crop


def test_square_crop_inside_image():
    img = np.full((100, 100, 3), 7, np.uint8)
    out = square_crop(img, 50, 50, 40, config.CROP_SIZE)
    assert out.shape == (config.CROP_SIZE, config.CROP_SIZE, 3)
    assert (out == 7).all()


def test_square_crop_outside_image_replicates_border():
    img = np.full((100, 100, 3), 7, np.uint8)
    out = square_crop(img, 0, 0, 60, config.CROP_SIZE)   # một phần nằm ngoài ảnh
    assert out.shape == (config.CROP_SIZE, config.CROP_SIZE, 3)
    assert (out == 7).all()                               # phần bù là nhân bản viền, không phải số 0


def test_features_have_expected_shape_and_hsv_histogram_sums_to_one():
    rng = np.random.default_rng(0)
    crops = rng.integers(0, 256, size=(3, config.CROP_SIZE, config.CROP_SIZE, 3), dtype=np.uint8)
    feats = extract_features(crops)
    assert feats.shape == (3, 1764 + 128)    # HOG + histogram HSV
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all()
    np.testing.assert_allclose(feats[:, -128:].sum(axis=1), 1.0, atol=1e-4)
