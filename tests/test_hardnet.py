import pytest
import cv2 as cv
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
from logging import Logger

from src.hardnet_descriptor import HardNet
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
    h, w = img.shape[:2]
    kp = [cv.KeyPoint(x=w // 2 + i * 20, y=h // 2 + i * 20, size=10) for i in range(5)]
    return {'kp': kp}


@pytest.fixture
def hardnet_instance(mock_logger):
    return HardNet("hardnet", mock_logger, config={'device': 'cpu'})


class TestHardNetRegistration:
    def test_registered_in_factory(self):
        assert "hardnet" in Descriptor._METHODS

    def test_factory_creation(self, mock_logger):
        obj = Descriptor.create("hardnet", mock_logger)
        assert isinstance(obj, HardNet)
        assert obj.default_norm == cv.NORM_L2

    def test_factory_creation_with_config(self, mock_logger):
        obj = Descriptor.create("hardnet", mock_logger,
                                config={'batch_size': 64, 'device': 'cpu'})
        assert isinstance(obj, HardNet)
        assert obj.batch_size == 64


class TestHardNetConfig:
    def test_default_patch_size(self, mock_logger):
        hn = HardNet("hardnet", mock_logger, config={})
        assert hn.patch_size == 32

    def test_custom_patch_size(self, mock_logger):
        hn = HardNet("hardnet", mock_logger, config={'patch_size': 64})
        assert hn.patch_size == 64

    def test_default_batch_size(self, mock_logger):
        hn = HardNet("hardnet", mock_logger, config={})
        assert hn.batch_size == 128

    def test_custom_device_cpu(self, mock_logger):
        hn = HardNet("hardnet", mock_logger, config={'device': 'cpu'})
        assert hn._device.type == 'cpu'

    def test_empty_config(self, mock_logger):
        hn = HardNet("hardnet", mock_logger, config={})
        assert hn.patch_size == 32
        assert hn.batch_size == 128

    def test_unknown_config_key_consumed(self, mock_logger):
        hn = HardNet("hardnet", mock_logger,
                     config={'unknown_key': 123, 'batch_size': 32})
        assert hn.batch_size == 32


class TestHardNetInference:
    def test_compute_returns_dict_structure(self, hardnet_instance, load_img, tmp_kp_features):
        result = hardnet_instance.compute(load_img("box.png"), tmp_kp_features)
        n_kp = len(tmp_kp_features['kp'])
        assert isinstance(result, dict)
        assert 'kp' in result
        assert 'des' in result
        assert len(result['kp']) == n_kp
        assert result['des'].shape == (n_kp, 128)
        assert result['des'].dtype == np.float32

    def test_compute_handles_empty_keypoints(self, mock_logger, random_img):
        hn = HardNet("hardnet", mock_logger, config={'device': 'cpu'})
        result = hn.compute(random_img, {'kp': []})

        assert len(result['kp']) == 0
        assert result['des'].shape[0] == 0

    def test_compute_filters_boundary_keypoints(self, mock_logger, random_img):
        gray_img = cv.cvtColor(random_img, cv.COLOR_BGR2GRAY)
        h, w = gray_img.shape

        boundary_kps = [
            cv.KeyPoint(x=0, y=0, size=10),
            cv.KeyPoint(x=w - 1, y=h - 1, size=10),
            cv.KeyPoint(x=w // 2, y=h // 2, size=10)
        ]
        hn = HardNet("hardnet", mock_logger, config={'device': 'cpu'})
        result = hn.compute(gray_img, {'kp': boundary_kps})
        assert len(result['kp']) == 1
        assert result['des'].shape == (1, 128)


class TestHardNetRobustness:
    def test_black_image(self, mock_logger):
        img = np.zeros((200, 200), dtype=np.uint8)

        hn = HardNet("hardnet", mock_logger, config={'device': 'cpu'})
        features = hn.compute(img, {'kp': []})

        assert isinstance(features, dict)
        assert 'kp' in features

    def test_very_small_image(self, mock_logger):
        tiny_img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)

        hn = HardNet("hardnet", mock_logger, config={'device': 'cpu'})
        try:
            features = hn.compute(tiny_img, {'kp': []})
            assert isinstance(features, dict)
        except Exception as e:
            pytest.fail(f"HardNet failed on tiny image: {e}")

    def test_high_noise_image(self, mock_logger):
        noise = np.random.randint(0, 2, (200, 200), dtype=np.uint8) * 255

        hn = HardNet("hardnet", mock_logger, config={'device': 'cpu'})
        features = hn.compute(noise, {'kp': [cv.KeyPoint(10, 10, 5)] * 5})

        assert isinstance(features, dict)
        assert 'des' in features

    def test_non_contiguous_array(self, mock_logger):
        img = np.random.randint(0, 255, (300, 300), dtype=np.uint8)[::2, ::2]

        hn = HardNet("hardnet", mock_logger, config={'device': 'cpu'})
        features = hn.compute(img, {'kp': []})
        assert isinstance(features, dict)
