import os
import pytest
import cv2 as cv
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
from logging import Logger

from src.descriptors import Descriptor
from src.detectors import Detector
from src.matchers import Matcher, BFMatcher, FLANNMatcher


@pytest.fixture
def mock_logger():
    return MagicMock(spec=Logger)


@pytest.fixture
def mock_descriptor():
    descriptor = MagicMock(spec=Descriptor)
    descriptor.default_norm = cv.NORM_L2
    return descriptor


@pytest.fixture
def mock_descriptor_hamming():
    descriptor = MagicMock(spec=Descriptor)
    descriptor.default_norm = cv.NORM_HAMMING
    return descriptor


@pytest.fixture
def test_features():
    np.random.seed(42)
    des1 = np.random.rand(50, 128).astype(np.float32)
    des2 = np.random.rand(60, 128).astype(np.float32)
    return {'des': des1}, {'des': des2}


@pytest.fixture
def test_features_binary():
    np.random.seed(42)
    des1 = np.random.randint(0, 255, (50, 32), dtype=np.uint8)
    des2 = np.random.randint(0, 255, (60, 32), dtype=np.uint8)
    return {'des': des1}, {'des': des2}


@pytest.fixture
def test_descriptors():
    np.random.seed(42)
    des1 = np.random.rand(50, 128).astype(np.float32)
    des2 = np.random.rand(60, 128).astype(np.float32)
    return des1, des2


@pytest.fixture
def identical_features():
    np.random.seed(42)
    des = np.random.rand(50, 128).astype(np.float32)
    return {'des': des}, {'des': des.copy()}


@pytest.fixture
def load_img():
    def _load(name, color=True):
        path = os.path.join(Path(__file__).parent.parent, "test_data", name)
        mode = cv.IMREAD_COLOR if color else cv.IMREAD_GRAYSCALE
        img = cv.imread(str(path), mode)
        if img is None:
            pytest.skip(f"Test image not found at {path}")
        return img

    return _load


@pytest.fixture
def get_kp(load_img, mock_logger):
    def _get(img_name, method_name="sift"):
        img = load_img(img_name)
        detector = Detector.create(method_name, mock_logger)
        features = detector.detect(img)
        return features.get('kp', [])
    return _get


@pytest.fixture
def get_descriptors(load_img, mock_logger):
    def _get(img_name, kp, method_name="sift"):
        img = load_img(img_name)
        descriptor = Descriptor.create(method_name, mock_logger)
        features = {'kp': kp}
        result = descriptor.compute(img, features)
        return result.get('des')

    return _get


class TestMatcherRegistry:
    def test_registration_completeness(self):
        expected_matchers = {"bf", "flann"}
        assert expected_matchers.issubset(Matcher._METHODS.keys())

    def test_internal_classes_not_registered(self):
        assert "matcher" not in Matcher._METHODS
        assert "opencvmatcher" not in Matcher._METHODS

    def test_factory_creation_types_bf(self, mock_logger, mock_descriptor):
        config = {'mode': 'simple'}
        matcher = Matcher.create("bf", mock_logger, config, mock_descriptor)
        assert isinstance(matcher, BFMatcher)
        assert isinstance(matcher, Matcher)

    def test_factory_creation_types_flann(self, mock_logger, mock_descriptor):
        config = {'mode': 'simple'}
        matcher = Matcher.create("flann", mock_logger, config, mock_descriptor)
        assert isinstance(matcher, FLANNMatcher)
        assert isinstance(matcher, Matcher)

    def test_factory_with_empty_config(self, mock_logger, mock_descriptor):
        matcher = Matcher.create("bf", mock_logger, {}, mock_descriptor)
        assert isinstance(matcher, BFMatcher)
        assert matcher.mode == 'simple'
        assert matcher.k == 1


class TestMatcherModes:
    def test_simple_mode_returns_list_of_dmatches(self, mock_logger, mock_descriptor, identical_features):
        config = {'mode': 'simple'}
        matcher = BFMatcher("bf", mock_logger, config, mock_descriptor)
        features1, features2 = identical_features
        result = matcher.match(features1, features2)

        assert isinstance(result, dict)
        assert 'matches' in result
        matches = result['matches']
        assert isinstance(matches, tuple)
        if matches:
            assert isinstance(matches[0], cv.DMatch)

    def test_knn_mode_returns_list_of_lists(self, mock_logger, mock_descriptor, identical_features):
        config = {'mode': 'knn', 'k': 2}
        matcher = BFMatcher("bf", mock_logger, config, mock_descriptor)
        features1, features2 = identical_features
        result = matcher.match(features1, features2)

        assert isinstance(result, dict)
        assert 'matches' in result
        matches = result['matches']
        assert isinstance(matches, tuple)
        if matches:
            assert isinstance(matches[0], tuple)
            assert isinstance(matches[0][0], cv.DMatch)

    def test_knn_returns_k_matches_per_query(self, mock_logger, mock_descriptor, identical_features):
        k_count = 3
        config = {'mode': 'knn', 'k': k_count}
        matcher = BFMatcher("bf", mock_logger, config, mock_descriptor)
        features1, features2 = identical_features
        result = matcher.match(features1, features2)
        matches = result['matches']

        if matches:
            assert len(matches[0]) == k_count


class TestBFMatcher:
    def test_bf_initialization_with_norm(self, mock_logger, mock_descriptor):
        config = {'mode': 'simple'}
        matcher = BFMatcher("bf", mock_logger, config, mock_descriptor)
        bf_matcher = matcher._init_matcher()
        assert isinstance(bf_matcher, cv.BFMatcher)

    def test_bf_simple_match_returns_valid_result(self, mock_logger, mock_descriptor, identical_features):
        config = {'mode': 'simple'}
        matcher = BFMatcher("bf", mock_logger, config, mock_descriptor)
        features1, features2 = identical_features
        result = matcher.match(features1, features2)

        assert isinstance(result, dict)
        assert 'matches' in result
        assert isinstance(result['matches'], tuple)

    def test_bf_knn_match_returns_valid_result(self, mock_logger, mock_descriptor, identical_features):
        config = {'mode': 'knn', 'k': 2}
        matcher = BFMatcher("bf", mock_logger, config, mock_descriptor)
        features1, features2 = identical_features
        result = matcher.match(features1, features2)

        assert isinstance(result, dict)
        assert 'matches' in result
        assert isinstance(result['matches'], tuple)

    def test_bf_reproducibility(self, mock_logger, mock_descriptor, test_features):
        config = {'mode': 'knn', 'k': 2}
        matcher = BFMatcher("bf", mock_logger, config, mock_descriptor)
        features1, features2 = test_features

        result1 = matcher.match(features1, features2)
        result2 = matcher.match(features1, features2)

        matches1 = result1['matches']
        matches2 = result2['matches']

        assert len(matches1) == len(matches2)
        if matches1 and matches2:
            assert matches1[0][0].queryIdx == matches2[0][0].queryIdx
            assert matches1[0][0].trainIdx == matches2[0][0].trainIdx


class TestFLANNMatcher:
    def test_flann_initialization_l2_norm(self, mock_logger, mock_descriptor):
        config = {'mode': 'simple'}
        matcher = FLANNMatcher("flann", mock_logger, config, mock_descriptor)
        assert matcher.index_params['algorithm'] == 1
        assert 'trees' in matcher.index_params
        assert matcher.index_params['trees'] == 5
        assert 'checks' in matcher.search_params
        assert matcher.search_params['checks'] == 50
