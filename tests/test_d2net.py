import pytest
import cv2 as cv
import numpy as np
import torch
from pathlib import Path
from unittest.mock import MagicMock
from logging import Logger

from src.d2net import D2Net
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
def d2net_instance(mock_logger):
    return D2Net("d2net", logger=mock_logger, config=None)


class TestD2NetRegistration:
    def test_registered_in_factory(self):
        assert "d2net" in Detector._METHODS
        assert "d2net" in Descriptor._METHODS

    def test_factory_creation(self, mock_logger):
        obj = Detector.create("d2net", mock_logger)
        assert isinstance(obj, D2Net)
        assert obj.default_norm == cv.NORM_L2

    def test_factory_creation_with_config(self, mock_logger):
        obj = Detector.create("d2net", mock_logger, config={'threshold': 0.01, 'use_relu': False})
        assert isinstance(obj, D2Net)
        assert obj._threshold == 0.01
        assert obj._use_relu is False


class TestD2NetConfig:
    def test_default_threshold(self, mock_logger):
        d2net = D2Net("d2net", mock_logger, config=None)
        assert d2net._threshold == 0.005

    def test_custom_threshold(self, mock_logger):
        d2net = D2Net("d2net", mock_logger, config={'threshold': 0.05})
        assert d2net._threshold == 0.05

    def test_custom_device(self, mock_logger):
        d2net = D2Net("d2net", mock_logger, config={'device': 'cpu'})
        assert d2net._device.type == 'cpu'

    def test_empty_config(self, mock_logger):
        d2net = D2Net("d2net", mock_logger, config={})
        assert d2net._threshold == 0.005

    def test_custom_config(self, mock_logger):
        d2net = D2Net(
            "d2net",
            mock_logger,
            config={
                'device': 'cpu',
                'threshold': 0.01,
                'use_relu': False
            }
        )
        assert d2net._threshold == 0.01
        assert d2net._use_relu is False
        assert d2net._device.type == 'cpu'

    def test_unknown_config_key_warns(self, mock_logger):
        D2Net("d2net", mock_logger, config={'unknown_key': 123})
        mock_logger.warning.assert_called()


class TestD2NetSingleton:
    def test_shared_model_weights(self, mock_logger):
        d2net1 = D2Net("d2net", logger=mock_logger, config=None)
        d2net2 = D2Net("d2net", logger=mock_logger, config=None)
        assert d2net1._model is d2net2._model

    def test_instance_parameter_independence(self, mock_logger):
        d2net_sensitive = D2Net("d2net1", mock_logger, config={'threshold': 0.001})
        d2net_strict = D2Net("d2net2", mock_logger, config={'threshold': 0.1})
        assert d2net_sensitive._threshold == 0.001
        assert d2net_strict._threshold == 0.1
        assert d2net_sensitive._threshold != d2net_strict._threshold

    def test_eval_mode_persistence(self, mock_logger):
        d2net = D2Net("d2net", mock_logger, config=None)
        assert not d2net._model.training

    def test_shared_model_after_deletion(self, mock_logger, random_img):
        d2net1 = D2Net("d2net1", mock_logger, config=None)
        d2net2 = D2Net("d2net2", mock_logger, config=None)
        del d2net1
        features = d2net2.detect(random_img)
        assert isinstance(features, dict)
        assert 'keypoints' in features

    def test_device_consistency(self, mock_logger):
        d2net = D2Net("d2net", mock_logger, config=None)
        model_device = next(d2net._model.parameters()).device
        assert d2net._device.type == model_device.type

    def test_different_thresholds_affect_keypoint_count(self, mock_logger, load_img):
        img = load_img("box.png")
        d2net_sensitive = D2Net("d2net_s", mock_logger, config={'threshold': 0.001})
        d2net_strict = D2Net("d2net_st", mock_logger, config={'threshold': 0.9})

        kp_sensitive = d2net_sensitive.detect(img).get('keypoints')
        kp_strict = d2net_strict.detect(img).get('keypoints')

        assert len(kp_sensitive) >= len(kp_strict)


class TestD2NetInference:
    def test_detect_returns_dict(self, d2net_instance, load_img):
        img = load_img("box.png")
        features = d2net_instance.detect(img)

        assert isinstance(features, dict)
        assert 'keypoints' in features
        assert 'descriptors' in features
        assert 'scores' in features

    def test_detect_returns_keypoints(self, d2net_instance, load_img):
        img = load_img("box.png")
        features = d2net_instance.detect(img)
        kp = features.get('keypoints')

        assert isinstance(kp, torch.Tensor)
        assert kp.ndim == 2
        assert kp.shape[1] == 2
        if len(kp) > 0:
            h, w = img.shape[:2]
            assert 0 <= kp[0][0] <= w
            assert 0 <= kp[0][1] <= h

    def test_detect_and_compute_consistency(self, d2net_instance, load_img):
        img = load_img("box.png")
        features = d2net_instance.detectAndCompute(img)

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
                assert des.shape[1] == 512
            else:
                assert des.dtype == np.float32
                assert des.shape[1] == 512

    def test_caching_mechanism(self, d2net_instance, load_img):
        img = load_img("box.png")
        detected = d2net_instance.detect(img)
        computed = d2net_instance.compute(img, detected)
        assert detected is computed

    def test_cache_invalidation_on_shape_change(self, d2net_instance, load_img):
        img = load_img("box.png")
        features1 = d2net_instance.detectAndCompute(img)

        img_resized = cv.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        features2 = d2net_instance.detectAndCompute(img_resized)

        assert features1.get('keypoints') is not features2.get('keypoints')
        assert features1.get('descriptors') is not features2.get('descriptors')

    def test_compute_with_internal_keypoints(self, d2net_instance, load_img):
        img = load_img("box.png")
        detected = d2net_instance.detect(img)
        computed = d2net_instance.compute(img, detected)

        assert len(computed.get('keypoints')) == len(detected.get('keypoints'))
        assert len(computed.get('descriptors')) == len(detected.get('keypoints'))

    def test_compute_with_external_keypoints(self, d2net_instance, load_img):
        img = load_img("box.png")
        fake_features = {'keypoints': np.array([[10, 10], [20, 20], [30, 30]])}
        features = d2net_instance.compute(img, fake_features)
        assert isinstance(features, dict)
        assert 'keypoints' in features

    def test_compute_after_detect_uses_cache_not_forward(self, d2net_instance, load_img, mock_logger):
        img = load_img("box.png")
        d2net_instance.detect(img)
        call_count_after_detect = mock_logger.info.call_count

        d2net_instance.compute(img, {})
        call_count_after_compute = mock_logger.info.call_count

        assert call_count_after_detect == call_count_after_compute


class TestSuperPointRobustness:
    def test_invalid_input_none(self, d2net_instance, mock_logger):
        features = d2net_instance._forward(None)
        assert features.get('keypoints') == ()
        assert features.get('descriptors') == ()
        mock_logger.error.assert_called_with("Input image is None. Detection aborted.")

    def test_black_image(self, d2net_instance, mock_logger):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        features = d2net_instance.detect(img)

        assert isinstance(features, dict)
        assert 'keypoints' in features
        if len(features.get('keypoints')) == 0:
            mock_logger.warning.assert_called()

    def test_compute_with_external_kp(self, d2net_instance, load_img):
        img = load_img("box.png")
        external = {'keypoints': np.array([[10, 10]])}
        features = d2net_instance.compute(img, external)
        assert isinstance(features, dict)
        assert 'keypoints' in features

    def test_very_small_image(self, d2net_instance):
        tiny_img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        try:
            features = d2net_instance.detect(tiny_img)
            assert isinstance(features, dict)
        except Exception as e:
            pytest.fail(f"D2Net failed on tiny image: {e}")

    def test_high_noise_image(self, d2net_instance):
        noise = np.random.randint(0, 2, (200, 200, 3), dtype=np.uint8) * 255
        features = d2net_instance.detect(noise)
        assert isinstance(features, dict)
        assert 'keypoints' in features

    def test_non_contiguous_array(self, d2net_instance):
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        features = d2net_instance.detect(img[::2, ::2, :])
        assert isinstance(features, dict)
        assert 'keypoints' in features
