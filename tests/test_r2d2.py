import pytest
import cv2 as cv
import numpy as np
import torch
from pathlib import Path
from unittest.mock import MagicMock
from logging import Logger

from src.r2d2 import R2D2
from src.detectors import Detector
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
def r2d2_instance(mock_logger):
    return R2D2("r2d2", logger=mock_logger, config=None)


class TestR2D2Registration:
    def test_registered_in_factory(self):
        assert "r2d2" in Detector._METHODS
        assert "r2d2" in Descriptor._METHODS

    def test_factory_creation(self, mock_logger):
        obj = Detector.create("r2d2", mock_logger)
        assert isinstance(obj, R2D2)
        assert obj.default_norm == cv.NORM_L2

    def test_factory_creation_with_config(self, mock_logger):
        obj = Detector.create("r2d2", mock_logger, config={'threshold': 0.01, 'top_k': 256})
        assert isinstance(obj, R2D2)
        assert obj._threshold == 0.01
        assert obj._top_k == 256


class TestR2D2Config:
    def test_default_threshold(self, mock_logger):
        r2d2 = R2D2("r2d2", mock_logger, config=None)
        assert r2d2._threshold == 0.005

    def test_custom_threshold(self, mock_logger):
        r2d2 = R2D2("r2d2", mock_logger, config={'threshold': 0.05})
        assert r2d2._threshold == 0.05

    def test_custom_device(self, mock_logger):
        r2d2 = R2D2("r2d2", mock_logger, config={'device': 'cpu'})
        assert r2d2._device.type == 'cpu'

    def test_empty_config(self, mock_logger):
        r2d2 = R2D2("r2d2", mock_logger, config={})
        assert r2d2._threshold == 0.005

    def test_custom_config(self, mock_logger):
        r2d2 = R2D2(
            "r2d2",
            mock_logger,
            config={
                'device': 'cpu',
                'threshold': 0.01,
                'top_k': 128
            }
        )
        assert r2d2._threshold == 0.01
        assert r2d2._top_k == 128
        assert r2d2._device.type == 'cpu'

class TestR2D2Singleton:
    def test_shared_model_weights(self, mock_logger):
        rd1 = R2D2("r2d2", logger=mock_logger, config=None)
        rd2 = R2D2("r2d2", logger=mock_logger, config=None)
        assert rd1._model is rd2._model

    def test_instance_parameter_independence(self, mock_logger):
        R2D2_sensitive = R2D2("rd1", mock_logger, config={'threshold': 0.001})
        R2D2_strict = R2D2("rd2", mock_logger, config={'threshold': 0.1})
        assert R2D2_sensitive._threshold == 0.001
        assert R2D2_strict._threshold == 0.1
        assert R2D2_sensitive._threshold != R2D2_strict._threshold

    def test_eval_mode_persistence(self, mock_logger):
        r2d2 = R2D2("r2d2", mock_logger, config=None)
        assert not r2d2._model.training

    def test_shared_model_after_deletion(self, mock_logger, random_img):
        rd1 = R2D2("rd1", mock_logger, config=None)
        rd2 = R2D2("rd2", mock_logger, config=None)
        del rd1
        features = rd2.detect(random_img)
        assert isinstance(features, dict)
        assert 'keypoints' in features

    def test_device_consistency(self, mock_logger):
        r2d2 = R2D2("r2d2", mock_logger, config=None)
        model_device = next(r2d2._model.parameters()).device
        assert r2d2._device.type == model_device.type

    def test_different_thresholds_affect_keypoint_count(self, mock_logger, load_img):
        img = load_img("box.png")
        r2d2_sensitive = R2D2("r2d2_s", mock_logger, config={'threshold': 0.001})
        r2d2_strict = R2D2("r2d2_st", mock_logger, config={'threshold': 0.9})

        kp_sensitive = r2d2_sensitive.detect(img).get('keypoints')
        kp_strict = r2d2_strict.detect(img).get('keypoints')

        assert len(kp_sensitive) >= len(kp_strict)


class TestR2D2Inference:
    def test_detect_returns_dict(self, r2d2_instance, load_img):
        img = load_img("box.png")
        features = r2d2_instance.detect(img)

        assert isinstance(features, dict)
        assert 'keypoints' in features
        assert 'descriptors' in features
        assert 'scores' in features

    def test_detect_returns_keypoints(self, r2d2_instance, load_img):
        img = load_img("box.png")
        features = r2d2_instance.detect(img)
        kp = features.get('keypoints')

        assert isinstance(kp, torch.Tensor)
        assert kp.ndim == 2
        assert kp.shape[1] == 2
        if len(kp) > 0:
            h, w = img.shape[:2]
            assert 0 <= kp[0][0] <= w
            assert 0 <= kp[0][1] <= h

    def test_detect_and_compute_consistency(self, r2d2_instance, load_img):
        img = load_img("box.png")
        features = r2d2_instance.detectAndCompute(img)

        kp = features.get('keypoints')
        des = features.get('descriptors')
        scores = features.get('scores')

        assert kp is not None
        assert des is not None
        assert scores is not None
        assert len(kp) == len(des)
        assert len(kp) == len(scores)
        if len(des) > 0:
            if isinstance(des, torch.Tensor):
                assert des.dtype == torch.float32
                assert des.shape[1] == 128
            else:
                assert des.dtype == np.float32
                assert des.shape[1] == 128

    def test_caching_mechanism(self, r2d2_instance, load_img):
        img = load_img("box.png")
        detected = r2d2_instance.detect(img)
        computed = r2d2_instance.compute(img, detected)
        assert detected is computed

    def test_cache_invalidation_on_shape_change(self, r2d2_instance, load_img):
        img = load_img("box.png")
        features1 = r2d2_instance.detectAndCompute(img)

        img_resized = cv.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        features2 = r2d2_instance.detectAndCompute(img_resized)

        assert features1.get('keypoints') is not features2.get('keypoints')
        assert features1.get('descriptors') is not features2.get('descriptors')

    def test_compute_with_internal_keypoints(self, r2d2_instance, load_img):
        img = load_img("box.png")
        detected = r2d2_instance.detect(img)
        computed = r2d2_instance.compute(img, detected)

        assert len(computed.get('keypoints')) == len(detected.get('keypoints'))
        assert len(computed.get('descriptors')) == len(detected.get('keypoints'))

    def test_compute_with_external_keypoints(self, r2d2_instance, load_img):
        img = load_img("box.png")
        fake_features = {'keypoints': np.array([[10, 10], [20, 20], [30, 30]])}
        features = r2d2_instance.compute(img, fake_features)
        assert isinstance(features, dict)
        assert 'keypoints' in features

    def test_compute_after_detect_uses_cache_not_forward(self, r2d2_instance, load_img, mock_logger):
        img = load_img("box.png")
        r2d2_instance.detect(img)
        call_count_after_detect = mock_logger.info.call_count

        r2d2_instance.compute(img, {})
        call_count_after_compute = mock_logger.info.call_count

        assert call_count_after_detect == call_count_after_compute


class TestR2D2Robustness:
    def test_invalid_input_none(self, r2d2_instance, mock_logger):
        features = r2d2_instance._forward(None)
        assert features.get('keypoints') == ()
        assert features.get('descriptors') == ()
        mock_logger.error.assert_called_with("Input image is None. Detection aborted.")

    def test_black_image(self, r2d2_instance, mock_logger):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        features = r2d2_instance.detect(img)

        assert isinstance(features, dict)
        assert 'keypoints' in features
        if len(features.get('keypoints')) == 0:
            mock_logger.warning.assert_called()

    def test_compute_with_external_kp(self, r2d2_instance, load_img):
        img = load_img("box.png")
        external = {'keypoints': np.array([[10, 10]])}
        features = r2d2_instance.compute(img, external)
        assert isinstance(features, dict)
        assert 'keypoints' in features

    def test_very_small_image(self, r2d2_instance):
        tiny_img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        try:
            features = r2d2_instance.detect(tiny_img)
            assert isinstance(features, dict)
        except Exception as e:
            pytest.fail(f"R2D2 failed on tiny image: {e}")

    def test_high_noise_image(self, r2d2_instance):
        noise = np.random.randint(0, 2, (200, 200, 3), dtype=np.uint8) * 255
        features = r2d2_instance.detect(noise)
        assert isinstance(features, dict)
        assert 'keypoints' in features

    def test_non_contiguous_array(self, r2d2_instance):
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        features = r2d2_instance.detect(img[::2, ::2, :])
        assert isinstance(features, dict)
        assert 'keypoints' in features
