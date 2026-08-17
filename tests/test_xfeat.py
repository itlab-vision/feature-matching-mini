import pytest
import cv2 as cv
import numpy as np
import torch
from pathlib import Path
from unittest.mock import MagicMock
from logging import Logger

from src.xfeat import XFeat
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
def xf_instance(mock_logger):
    return XFeat("xfeat", logger=mock_logger, config=None)


class TestXFeatRegistration:
    def test_registered_in_factory(self):
        assert "xfeat" in Detector._METHODS
        assert "xfeat" in Descriptor._METHODS

    def test_factory_creation(self, mock_logger):
        obj = Detector.create("xfeat", mock_logger)
        assert isinstance(obj, XFeat)
        assert obj.default_norm == cv.NORM_L2

    def test_factory_creation_with_config(self, mock_logger):
        obj = Detector.create("xfeat", mock_logger, config={'threshold': 0.01, 'top_k': 256})
        assert isinstance(obj, XFeat)
        assert obj._threshold == 0.01
        assert obj._top_k == 256


class TestXFeatConfig:
    def test_default_threshold(self, mock_logger):
        xf = XFeat("xfeat", mock_logger, config=None)
        assert xf._threshold == 0.005

    def test_custom_threshold(self, mock_logger):
        xf = XFeat("xfeat", mock_logger, config={'threshold': 0.05})
        assert xf._threshold == 0.05

    def test_custom_device(self, mock_logger):
        xf = XFeat("xfeat", mock_logger, config={'device': 'cpu'})
        assert xf._device.type == 'cpu'

    def test_custom_top_k(self, mock_logger):
        xf = XFeat("xfeat", mock_logger, config={'top_k': 256})
        assert xf._top_k == 256

    def test_empty_config(self, mock_logger):
        xf = XFeat("xfeat", mock_logger, config={})
        assert xf._threshold == 0.005

    def test_unknown_config_key_warns(self, mock_logger):
        XFeat("xfeat", mock_logger, config={'unknown_key': 123})
        mock_logger.warning.assert_called()


class TestXFeatSingleton:
    def test_shared_model_weights(self, mock_logger):
        xf1 = XFeat("xfeat", logger=mock_logger, config=None)
        xf2 = XFeat("xfeat", logger=mock_logger, config=None)
        assert xf1._model is xf2._model

    def test_instance_parameter_independence(self, mock_logger):
        xf_sensitive = XFeat("xf1", mock_logger, config={'threshold': 0.001, 'top_k': 128})
        xf_strict = XFeat("xf2", mock_logger, config={'threshold': 0.1, 'top_k': 256})
        assert xf_sensitive._threshold == 0.001
        assert xf_strict._threshold == 0.1
        assert xf_sensitive._top_k == 128
        assert xf_strict._top_k == 256
        assert xf_sensitive._threshold != xf_strict._threshold
        assert xf_sensitive._top_k != xf_strict._top_k

    def test_eval_mode_persistence(self, mock_logger):
        xf = XFeat("xf", mock_logger, config=None)
        assert not xf._model.training

    def test_shared_model_after_deletion(self, mock_logger, random_img):
        xf1 = XFeat("xf1", mock_logger, config=None)
        xf2 = XFeat("xf2", mock_logger, config=None)
        del xf1
        features = xf2.detect(random_img)
        assert isinstance(features, dict)
        assert 'keypoints' in features

    def test_device_consistency(self, mock_logger):
        xf = XFeat("xf", mock_logger, config=None)
        model_device = next(xf._model.parameters()).device
        assert xf._device.type == model_device.type

    def test_different_thresholds_affect_keypoint_count(self, mock_logger, load_img):
        img = load_img("box.png")
        import logging
        logger = logging.getLogger('xfeat')

        xf_sensitive = XFeat("xf_s", logger, config={'threshold': 0.001, 'top_k': 32})
        xf_strict = XFeat("xf_st", logger, config={'threshold': 0.06, 'top_k': 4})

        kp_sensitive = xf_sensitive.detect(img).get('keypoints')
        kp_strict = xf_strict.detect(img).get('keypoints')

        assert len(kp_sensitive) >= len(kp_strict)
        assert len(kp_sensitive) == 32
        assert len(kp_strict) == 4


class TestXFeatInference:
    def test_detect_returns_dict(self, xf_instance, load_img):
        img = load_img("box.png")
        features = xf_instance.detect(img)

        assert isinstance(features, dict)
        assert 'keypoints' in features
        assert 'descriptors' in features
        assert 'scores' in features

    def test_detect_returns_keypoints(self, xf_instance, load_img):
        img = load_img("box.png")
        features = xf_instance.detect(img)
        kp = features.get('keypoints')

        assert isinstance(kp, torch.Tensor)
        assert kp.ndim == 2
        assert kp.shape[1] == 2
        if len(kp) > 0:
            h, w = img.shape[:2]
            assert 0 <= kp[0][0] <= w
            assert 0 <= kp[0][1] <= h

    def test_detect_and_compute_consistency(self, xf_instance, load_img):
        img = load_img("box.png")
        features = xf_instance.detectAndCompute(img)

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
                assert des.shape[1] == 64
            else:
                assert des.dtype == np.float32
                assert des.shape[1] == 64

    def test_caching_mechanism(self, xf_instance, load_img):
        img = load_img("box.png")
        detected = xf_instance.detect(img)
        computed = xf_instance.compute(img, detected)
        assert detected is computed

    def test_cache_invalidation_on_shape_change(self, xf_instance, load_img):
        img = load_img("box.png")
        features1 = xf_instance.detectAndCompute(img)

        img_resized = cv.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        features2 = xf_instance.detectAndCompute(img_resized)

        assert features1.get('keypoints') is not features2.get('keypoints')
        assert features1.get('descriptors') is not features2.get('descriptors')

    def test_compute_with_internal_keypoints(self, xf_instance, load_img):
        img = load_img("box.png")
        detected = xf_instance.detect(img)
        computed = xf_instance.compute(img, detected)

        assert len(computed.get('keypoints')) == len(detected.get('keypoints'))
        assert len(computed.get('descriptors')) == len(detected.get('keypoints'))

    def test_compute_with_external_keypoints(self, xf_instance, load_img):
        img = load_img("box.png")
        fake_features = {'keypoints': np.array([[10, 10], [20, 20], [30, 30]])}
        features = xf_instance.compute(img, fake_features)
        assert isinstance(features, dict)
        assert 'keypoints' in features

    def test_compute_after_detect_uses_cache_not_forward(self, xf_instance, load_img, mock_logger):
        img = load_img("box.png")
        xf_instance.detect(img)
        call_count_after_detect = mock_logger.info.call_count

        xf_instance.compute(img, {})
        call_count_after_compute = mock_logger.info.call_count

        assert call_count_after_detect == call_count_after_compute


class TestXFeatRobustness:
    def test_invalid_input_none(self, xf_instance, mock_logger):
        features = xf_instance._forward(None)
        assert features.get('keypoints') == ()
        assert features.get('descriptors') == ()
        mock_logger.error.assert_called_with("Input image is None. Detection aborted.")

    def test_black_image(self, xf_instance, mock_logger):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        features = xf_instance.detect(img)

        assert isinstance(features, dict)
        assert 'keypoints' in features
        if len(features.get('keypoints')) == 0:
            mock_logger.warning.assert_called()

    def test_compute_with_external_kp(self, xf_instance, load_img):
        img = load_img("box.png")
        external = {'keypoints': np.array([[10, 10]])}
        features = xf_instance.compute(img, external)
        assert isinstance(features, dict)
        assert 'keypoints' in features

    def test_very_small_image(self, xf_instance):
        tiny_img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        try:
            features = xf_instance.detect(tiny_img)
            assert isinstance(features, dict)
        except Exception as e:
            pytest.fail(f"XFeat failed on tiny image: {e}")

    def test_high_noise_image(self, xf_instance):
        noise = np.random.randint(0, 2, (200, 200, 3), dtype=np.uint8) * 255
        features = xf_instance.detect(noise)
        assert isinstance(features, dict)
        assert 'keypoints' in features

    def test_non_contiguous_array(self, xf_instance):
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        features = xf_instance.detect(img[::2, ::2, :])
        assert isinstance(features, dict)
        assert 'keypoints' in features
