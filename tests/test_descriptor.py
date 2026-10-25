import pytest
import cv2 as cv
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
from logging import Logger

from src.image_utils import read_image
from src.algorithms import DNN_ALGORITHMS, ALL_DETECTORS, ALL_DESCRIPTORS
import src.backend.opencv.model_loader  # noqa: F401
import src.backend.opencv.inference_api  # noqa: F401
import src.backend.opencv.io_adapter  # noqa: F401
from src.detectors import Detector
from src.descriptors import Descriptor, SIFTDescriptor, OpenCVDescriptor
from src.feature_matcher import FeatureMatcherCV2


@pytest.fixture
def mock_logger():
    return MagicMock(spec=Logger)


@pytest.fixture
def load_img():
    def _load(name, input_type='numpy'):
        path = Path(__file__).parent.parent / "test_data" / name
        return read_image(path)
    return _load


@pytest.fixture
def get_kp(load_img, mock_logger):
    def _get(img_name, method_name="sift", input_type='numpy', config=None):
        img = load_img(img_name, input_type=input_type)
        detector = Detector.create(method_name, mock_logger, config=config)
        features = detector.detect(img)
        return features
    return _get


class TestDescriptorRegistry:
    def test_registration_completeness(self):
        expected = {"sift", "orb", "akaze", "brisk", "kaze"}
        assert expected.issubset(Descriptor._METHODS.keys())

    def test_internal_classes_not_registered(self):
        assert "opencv" not in Descriptor._METHODS
        assert "descriptor" not in Descriptor._METHODS

    def test_factory_creation_types(self, mock_logger):
        obj = Descriptor.create("sift", mock_logger)
        assert isinstance(obj, SIFTDescriptor)
        assert isinstance(obj, OpenCVDescriptor)


class TestDescriptorFactory:
    def test_factory_creation_types(self, mock_logger):
        obj = Descriptor.create("sift", mock_logger)
        assert isinstance(obj, SIFTDescriptor)
        assert isinstance(obj, OpenCVDescriptor)

    def test_create_unknown_descriptor_raises_error(self, mock_logger):
        with pytest.raises(ValueError, match="Descriptor 'unknown' not found"):
            Descriptor.create("unknown", mock_logger)

    def test_empty_config(self, mock_logger):
        obj = Descriptor.create("sift", mock_logger, config={})
        assert isinstance(obj, SIFTDescriptor)

    def test_none_config(self, mock_logger):
        obj = Descriptor.create("sift", mock_logger, config=None)
        assert isinstance(obj, SIFTDescriptor)

    def test_nfeatures_config(self, mock_logger, load_img, get_kp):
        img = load_img("box.png")
        limit = 10
        features = get_kp("box.png", "sift")
        descriptor = Descriptor.create("sift", mock_logger, config={'nfeatures': limit})
        features = descriptor.compute(img, features)
        assert features.get('descriptors') is not None

    def test_config_isolation(self, mock_logger, load_img, get_kp):
        img = load_img("box.png")
        features = get_kp("box.png", "sift")

        desc_default = Descriptor.create("sift", mock_logger)
        desc_limited = Descriptor.create("sift", mock_logger, config={'nfeatures': 5})

        features_default = desc_default.compute(img, {'keypoints': features.get('keypoints')})
        features_limited = desc_limited.compute(img, {'keypoints': features.get('keypoints')})

        assert features_default.get('descriptors') is not None
        assert features_limited.get('descriptors') is not None


ALGORITHMS_CPU_ONLY = {'doghardnet_lightglue'}


class TestDescriptorCompute:
    @pytest.mark.parametrize("detector_name", ALL_DETECTORS)
    @pytest.mark.parametrize("descriptor_name", ALL_DESCRIPTORS)
    def test_all_methods_compute_descriptors(self, detector_name, descriptor_name, mock_logger, load_img, get_kp):
        if (detector_name in FeatureMatcherCV2._DETECTOR_DESCRIPTOR_COMPATIBILITY
                and descriptor_name in FeatureMatcherCV2._DETECTOR_DESCRIPTOR_COMPATIBILITY[detector_name]):
            is_neural = detector_name in DNN_ALGORITHMS

            if is_neural:
                img = load_img("box.png", input_type='tensor')
                config = {'device': 'cpu'} if detector_name in ALGORITHMS_CPU_ONLY else {}
                features = get_kp("box.png", detector_name, input_type='tensor', config=config)

                assert isinstance(features, dict)
                assert 'keypoints' in features or 'keypoints' in features
            else:
                img = load_img("box.png")
                features = get_kp("box.png", detector_name)

                descriptor = Descriptor.create(descriptor_name, mock_logger)
                features = descriptor.compute(img, features)

                assert isinstance(features.get('keypoints'), (tuple, list, np.ndarray))
                if features.get('descriptors') is not None:
                    assert isinstance(features.get('descriptors'), np.ndarray)
                    assert len(features.get('descriptors')) == len(features.get('keypoints'))

                assert mock_logger.info.called

    def test_sift_descriptor_size(self, mock_logger, load_img, get_kp):
        img = load_img("box.png")
        descriptor = Descriptor.create("sift", mock_logger)
        features = get_kp("box.png", "sift")
        features['keypoints'] = features.get('keypoints')[:5]
        features = descriptor.compute(img, features)

        assert features.get('descriptors') is not None
        assert features.get('descriptors').shape[1] == 128

    def test_orb_descriptor_size(self, mock_logger, load_img, get_kp):
        img = load_img("box.png")
        descriptor = Descriptor.create("orb", mock_logger)
        features = get_kp("box.png", "orb")
        features['keypoints'] = features.get('keypoints')[:5]
        features = descriptor.compute(img, features)

        assert features.get('descriptors') is not None
        assert features.get('descriptors').shape[1] == 32

    def test_beblid_scale_factor_config(self, mock_logger, load_img, get_kp):
        img = load_img("box.png")
        features = get_kp("box.png", "orb")
        features['keypoints'] = features.get('keypoints')[:10]
        descriptor = Descriptor.create("beblid", mock_logger, config={'scale_factor': 1.0})
        features = descriptor.compute(img, features)
        assert features.get('descriptors') is not None

    def test_teblid_scale_factor_config(self, mock_logger, load_img, get_kp):
        img = load_img("box.png")
        features = get_kp("box.png", "orb")
        features['keypoints'] = features.get('keypoints')[:10]
        descriptor = Descriptor.create("teblid", mock_logger, config={'scale_factor': 1.0})
        features = descriptor.compute(img, features)
        assert features.get('descriptors') is not None

    def test_invalid_input_none_image(self, mock_logger, get_kp):
        descriptor = Descriptor.create("sift", mock_logger)
        features = get_kp("box.png", "sift")
        features['keypoints'] = features.get('keypoints')[:5]
        features = descriptor.compute(None, features)

        assert features.get('descriptors') == ()
        assert mock_logger.error.called

    def test_empty_keypoints_warning(self, mock_logger, load_img):
        img = load_img("box.png")
        descriptor = Descriptor.create("orb", mock_logger)
        features = descriptor.compute(img, {'keypoints': ()})

        assert features.get('descriptors') is None or len(features.get('descriptors')) == 0
        assert mock_logger.warning.called

    def test_reproducibility(self, mock_logger, load_img, get_kp):
        img = load_img("box.png")
        descriptor = Descriptor.create("sift", mock_logger)
        features = get_kp("box.png", "sift")
        features['keypoints'] = features.get('keypoints')[:10]

        features1 = descriptor.compute(img, features)
        features2 = descriptor.compute(img, features)
        assert np.array_equal(features1.get('descriptors'), features2.get('descriptors'))

    def test_compare_descriptors_on_different_images(self, mock_logger, load_img, get_kp):
        descriptor = Descriptor.create("sift", mock_logger)

        img_box = load_img("box.png")
        features_box = get_kp("box.png")
        features_box = descriptor.compute(img_box, features_box)

        img_scene = load_img("box_in_scene.png")
        features_scene = get_kp("box_in_scene.png")
        features_scene = descriptor.compute(img_scene, features_scene)

        assert features_box.get('descriptors') is not None
        assert features_scene.get('descriptors') is not None
        assert len(features_scene.get('descriptors')) > len(features_box.get('descriptors'))

    def test_descriptor_consistency(self, mock_logger, load_img, get_kp):
        img = load_img("box.png")
        features = get_kp("box.png", "orb")
        features['keypoints'] = features.get('keypoints')[:10]

        descriptor = Descriptor.create("orb", mock_logger)
        features1 = descriptor.compute(img, features)
        features2 = descriptor.compute(img, features)
        assert np.array_equal(features1.get('descriptors'), features2.get('descriptors'))

    def test_invalid_input_none(self, mock_logger, get_kp):
        descriptor = Descriptor.create("sift", mock_logger)
        features = get_kp("box.png", "sift")

        features = descriptor.compute(None, features)
        assert features.get('descriptors') == ()
        mock_logger.error.assert_called()


class TestDescriptorRobustness:
    def test_invalid_input_none(self, mock_logger, get_kp):
        descriptor = Descriptor.create("sift", mock_logger)
        features = descriptor.compute(None, get_kp("box.png", "sift"))
        features['keypoints'] = features.get('keypoints')[:5]
        assert features.get('descriptors') == ()
        assert mock_logger.error.called

    def test_keypoints_outside_bounds(self, mock_logger, load_img):
        img = load_img("box.png")
        descriptor = Descriptor.create("sift", mock_logger)
        bad_kp = {'keypoints': [cv.KeyPoint(x=10000, y=10000, size=10)]}
        features = descriptor.compute(img, bad_kp)
        if features.get('descriptors') is not None and len(features.get('descriptors')) > 0:
            assert np.all(features.get('descriptors') == 0) or len(features.get('keypoints')) == 0
        else:
            assert len(features.get('keypoints')) == 0 or features.get('descriptors') is None

    def test_empty_keypoints_warning(self, mock_logger, load_img):
        img = load_img("box.png")
        descriptor = Descriptor.create("orb", mock_logger)
        features = descriptor.compute(img, {'keypoints': ()})
        assert mock_logger.warning.called
        assert features.get('descriptors') is None


class TestDescriptorInvariance:
    def test_brightness_invariance(self, mock_logger, load_img, get_kp):
        img = load_img("box.png")
        bright_img = cv.convertScaleAbs(img, alpha=1.2, beta=30)
        descriptor = Descriptor.create("sift", mock_logger)
        features = get_kp("box.png", "sift")
        features['keypoints'] = features.get('keypoints')[:10]
        features1 = descriptor.compute(img, features)
        features2 = descriptor.compute(bright_img, features)
        cos_sim = (np.dot(features1.get('descriptors')[0], features2.get('descriptors')[0])
                   / (np.linalg.norm(features1.get('descriptors')[0])
                      * np.linalg.norm(features2.get('descriptors')[0])))
        assert cos_sim > 0.95

    def test_flip_consistency(self, mock_logger, load_img):
        img = load_img("box.png")
        flipped_img = cv.flip(img, 1)
        descriptor = Descriptor.create("orb", mock_logger)
        features = {'keypoints': [cv.KeyPoint(100, 100, 10)]}
        features_flipped = {'keypoints': [cv.KeyPoint(img.shape[1] - 100, 100, 10)]}
        features = descriptor.compute(img, features)
        features_flipped = descriptor.compute(flipped_img, features_flipped)
        assert features.get('descriptors').shape == features_flipped.get('descriptors').shape
