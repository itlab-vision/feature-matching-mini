import pytest
import cv2 as cv
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
from logging import Logger

from src.tfeat_descriptor import TFeat
from src.descriptors import Descriptor


@pytest.fixture
def mock_logger():
    return MagicMock(spec=Logger)


@pytest.fixture
def load_img():
    def _load(name, color=True):
        path = Path(__file__).parent.parent / "test_data" / name
        mode = cv.IMREAD_COLOR if color else cv.IMREAD_GRAYSCALE
        img = cv.imread(str(path), mode)
        if img is None:
            pytest.skip(f"Test image not found at {path}")
        return img

    return _load


@pytest.fixture
def random_img():
    return np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)


@pytest.fixture
def tmp_kp_features(load_img):
    img = load_img("box.png")
    detector = cv.SIFT_create(nfeatures=10)
    kp = detector.detect(img, None)
    return {'kp': kp}


@pytest.fixture
def tfeat_instance(mock_logger):
    return TFeat("tfeat", mock_logger, config={'device': 'cpu'})


class TestTFeatRegistration:
    def test_registered_in_factory(self):
        assert "tfeat" in Descriptor._METHODS

    def test_factory_creation(self, mock_logger):
        obj = Descriptor.create("tfeat", mock_logger)
        assert isinstance(obj, TFeat)
        assert obj.default_norm == cv.NORM_L2

    def test_factory_creation_with_config(self, mock_logger):
        obj = Descriptor.create("tfeat", mock_logger, config={'magfactor': 5, 'device': 'cpu'})
        assert isinstance(obj, TFeat)
        assert obj.mag_factor == 5


class TestTFeatConfig:
    def test_default_mag_factor(self, mock_logger):
        tfeat = TFeat("tfeat", mock_logger, config={})
        assert tfeat.mag_factor == 3

    def test_custom_mag_factor(self, mock_logger):
        tfeat = TFeat("tfeat", mock_logger, config={'magfactor': 7})
        assert tfeat.mag_factor == 7

    def test_custom_device_cpu(self, mock_logger):
        tfeat = TFeat("tfeat", mock_logger, config={'device': 'cpu'})
        assert tfeat._device.type == 'cpu'

    def test_empty_config(self, mock_logger):
        tfeat = TFeat("tfeat", mock_logger, config={})
        assert tfeat.mag_factor == 3

    def test_unknown_config_key_consumed(self, mock_logger):
        tfeat = TFeat("tfeat", mock_logger, config={'unknown_key': 123, 'magfactor': 4})
        assert tfeat.mag_factor == 4


class TestTFeatInference:
    def test_compute_returns_dict_structure(self, tfeat_instance, load_img, tmp_kp_features):
        result = tfeat_instance.compute(load_img("box.png"), tmp_kp_features)
        n_kp = len(tmp_kp_features['kp'])
        assert isinstance(result, dict)
        assert 'kp' in result
        assert 'des' in result
        assert len(result['kp']) == n_kp
        assert result['des'].shape == (n_kp, 128)
        assert result['des'].dtype == np.float32

    def test_compute_handles_empty_keypoints(self, mock_logger, random_img):
        tfeat = TFeat("tfeat", mock_logger, config={'device': 'cpu'})
        result = tfeat.compute(random_img, {'kp': []})

        assert len(result['kp']) == 0
        assert result['des'].shape[0] == 0


class TestTFeatRobustness:
    def test_black_image(self, mock_logger):
        img = np.zeros((200, 200), dtype=np.uint8)

        tfeat = TFeat("tfeat", mock_logger, config={'device': 'cpu'})
        features = tfeat.compute(img, {'kp': []})

        assert isinstance(features, dict)
        assert 'kp' in features

    def test_very_small_image(self, mock_logger):
        tiny_img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)

        tfeat = TFeat("tfeat", mock_logger, config={'device': 'cpu'})
        try:
            features = tfeat.compute(tiny_img, {'kp': []})
            assert isinstance(features, dict)
        except Exception as e:
            pytest.fail(f"TFeat failed on tiny image: {e}")

    def test_high_noise_image(self, mock_logger):
        noise = np.random.randint(0, 2, (200, 200), dtype=np.uint8) * 255

        tfeat = TFeat("tfeat", mock_logger, config={'device': 'cpu'})
        features = tfeat.compute(noise, {'kp': [cv.KeyPoint(10, 10, 5)] * 5})

        assert isinstance(features, dict)
        assert 'des' in features

    def test_non_contiguous_array(self, mock_logger):
        img = np.random.randint(0, 255, (300, 300), dtype=np.uint8)[::2, ::2]

        tfeat = TFeat("tfeat", mock_logger, config={'device': 'cpu'})
        features = tfeat.compute(img, {'kp': []})
        assert isinstance(features, dict)
