import pytest
import cv2 as cv
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
from logging import Logger

from src.detectors import Detector, SIFTDetector, OpenCVDetector
from src.lightglue_pipeline import LightGlueFeatureExtractor  # noqa: F401
from src.super_point import SuperPoint  # noqa: F401

from src.algorithms import DNN_ALGORITHMS, EXECUTORCH_ALGORITHMS
from src.image_utils import read_image


@pytest.fixture
def mock_logger():
    return MagicMock(spec=Logger)


@pytest.fixture
def load_img():
    def _load(name, input_type='numpy'):
        path = Path(__file__).parent.parent / "test_data" / name
        return read_image(path, input_type=input_type)
    return _load


class TestDetectorRegistry:
    def test_registration_completeness(self):
        expected_algos = {"sift", "orb", "akaze", "fast", "brisk", "kaze"}
        assert expected_algos.issubset(Detector._METHODS.keys())

    def test_internal_classes_not_registered(self):
        assert "opencv" not in Detector._METHODS
        assert "detector" not in Detector._METHODS

    def test_factory_creation_types(self, mock_logger):
        detector = Detector.create("sift", mock_logger)
        assert isinstance(detector, SIFTDetector)
        assert isinstance(detector, OpenCVDetector)


class TestDetectorFactory:
    def test_factory_creation_types(self, mock_logger):
        detector = Detector.create("sift", mock_logger)
        assert isinstance(detector, SIFTDetector)
        assert isinstance(detector, OpenCVDetector)

    def test_create_unknown_detector_raises_error(self, mock_logger):
        with pytest.raises(ValueError, match="Detector 'unknown' not found"):
            Detector.create("unknown", mock_logger)

    def test_parameters_passed_to_cv2(self, mock_logger, load_img):
        img = load_img("box.png")
        limit = 10
        detector = Detector.create("sift", mock_logger, config={'nfeatures': limit})
        features = detector.detect(img)
        assert len(features.get('kp')) <= limit + 5

    def test_empty_config(self, mock_logger):
        detector = Detector.create("sift", mock_logger, config={})
        assert isinstance(detector, SIFTDetector)

    def test_none_config(self, mock_logger):
        detector = Detector.create("sift", mock_logger, config=None)
        assert isinstance(detector, SIFTDetector)


ALGORITHMS_CPU_ONLY = {'doghardnet_lightglue'}


class TestDetectorDetect:
    @pytest.mark.parametrize("method_name", Detector._METHODS.keys())
    def test_all_methods_return_valid_tuple(self, method_name, mock_logger, load_img):
        if method_name in EXECUTORCH_ALGORITHMS:
            pytest.skip("Model files in the torchexecut format are missing,"
                        " so we are skipping these combinations")

        if method_name in DNN_ALGORITHMS:
            img = load_img("box.png", input_type='tensor')

            if method_name in ALGORITHMS_CPU_ONLY:
                config = {'device': 'cpu'}
            else:
                config = {}

            detector = Detector.create(method_name, mock_logger, config=config)
        else:
            img = load_img("box.png")
            detector = Detector.create(method_name, mock_logger)

        features = detector.detect(img)

        if method_name in DNN_ALGORITHMS:
            assert 'keypoints' in features
            assert features.get('keypoints') is not None
        else:
            kp = features.get('kp')
            assert isinstance(kp, (tuple, list))
            if kp:
                assert isinstance(kp[0], cv.KeyPoint)

    def test_reproducibility(self, mock_logger, load_img):
        img = load_img("box.png")
        detector = Detector.create("orb", mock_logger)

        features1 = detector.detect(img)
        features2 = detector.detect(img)

        assert len(features1.get('kp')) == len(features2.get('kp'))
        assert features1.get('kp')[0].pt == features2.get('kp')[0].pt

    def test_orb_nfeatures_parameter(self, mock_logger, load_img):
        img = load_img("box_in_scene.png")
        limit = 30

        detector = Detector.create("orb", mock_logger, config={'nfeatures': limit})
        features = detector.detect(img)

        assert len(features.get('kp')) <= limit

    def test_sift_nfeatures_parameter(self, mock_logger, load_img):
        img = load_img("box_in_scene.png")
        limit = 20

        detector = Detector.create("sift", mock_logger, config={'nfeatures': limit})
        features = detector.detect(img)

        assert len(features.get('kp')) <= limit

    def test_empty_image_logging(self, mock_logger):
        detector = Detector.create("sift", mock_logger)
        black_img = np.zeros((100, 100), dtype=np.uint8)

        detector.detect(black_img)
        mock_logger.warning.assert_called()

    def test_compare_box_and_scene(self, mock_logger, load_img):
        detector = Detector.create("sift", mock_logger)

        features_box = detector.detect(load_img("box.png"))
        features_scene = detector.detect(load_img("box_in_scene.png"))
        assert len(features_box.get('kp')) < len(features_scene.get('kp'))

    def test_invalid_input_none(self, mock_logger):
        detector = Detector.create("sift", mock_logger)
        features = detector.detect(None)

        assert features.get('kp') == ()
        assert mock_logger.error.called


class TestDetectorRobustness:
    def test_all_white_image(self, mock_logger):
        white_img = np.ones((200, 200), dtype=np.uint8) * 255
        detector = Detector.create("sift", mock_logger)
        features = detector.detect(white_img)
        assert len(features.get('kp')) == 0

    def test_very_small_image(self, mock_logger):
        tiny_img = np.zeros((5, 5), dtype=np.uint8)
        detector = Detector.create("orb", mock_logger)
        features = detector.detect(tiny_img)
        assert isinstance(features.get('kp'), tuple)

    def test_color_image_support(self, mock_logger, load_img):
        path = Path(__file__).parent.parent / "test_data" / "box.png"
        img_bgr = cv.imread(str(path))
        if img_bgr is None:
            pytest.skip("Image not found")

        detector = Detector.create("sift", mock_logger)
        features = detector.detect(img_bgr)
        assert len(features.get('kp')) > 0

    def test_invalid_config_key_raises(self, mock_logger):
        with pytest.raises((TypeError, cv.error)):
            Detector.create("sift", mock_logger, config={'invalid_param': 999})


class TestKeypointProperties:
    @pytest.mark.parametrize("method_name", ["sift", "orb"])
    def test_keypoints_within_image_bounds(self, method_name, mock_logger, load_img):
        img = load_img("box.png")
        h, w = img.shape[:2]
        detector = Detector.create(method_name, mock_logger)
        features = detector.detect(img)
        for p in features.get('kp'):
            x, y = p.pt
            assert 0 <= x < w
            assert 0 <= y < h


class TestDetectorFunctional:
    def test_detectors_uniqueness(self, mock_logger, load_img):
        img = load_img("box.png")
        features_sift = Detector.create("sift", mock_logger).detect(img)
        features_orb = Detector.create("orb", mock_logger).detect(img)
        assert features_sift.get('kp')[0].pt != features_orb.get('kp')[0].pt

    def test_scale_impact(self, mock_logger, load_img):
        img = load_img("box.png")
        detector = Detector.create("sift", mock_logger)
        features_original = detector.detect(img)
        small_img = cv.resize(img, (0, 0), fx=0.5, fy=0.5)
        features_small = detector.detect(small_img)
        assert len(features_original.get('kp')) > len(features_small.get('kp'))

    def test_config_isolation(self, mock_logger, load_img):
        img = load_img("box.png")
        features_default = Detector.create("sift", mock_logger).detect(img)
        features_limited = Detector.create("sift", mock_logger, config={'nfeatures': 5}).detect(img)
        assert len(features_limited.get('kp')) < len(features_default.get('kp'))
