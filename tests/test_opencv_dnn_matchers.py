import pytest
import cv2 as cv
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
from logging import Logger

from src.opencv_dnn_extractors import DISKOpenCV
from src.opencv_dnn_matchers import LightGlueOpenCVMatcher
from src.matchers import Matcher


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
def tmp_features(load_img):
    img = load_img("box.png")
    h, w = img.shape[:2]
    kp = [cv.KeyPoint(x=w // 2 + i * 15, y=h // 2 + i * 15, size=10) for i in range(10)]
    des = np.random.rand(10, 128).astype(np.float32)
    return {
        'kp': kp,
        'des': des,
        'img_shape': img.shape
    }


class TestLightGlueRegistration:
    def test_registered_in_factory(self, mock_logger):
        extractor = DISKOpenCV("disk", mock_logger, config={})
        obj = Matcher.create("lightglueopencv", mock_logger, descriptor_name=extractor, config={})
        assert isinstance(obj, LightGlueOpenCVMatcher)
        assert "lightglueopencv" in Matcher._METHODS


class TestLightGlueConfig:
    def test_default_score_threshold(self, mock_logger):
        extractor = DISKOpenCV("disk", mock_logger, config={})
        matcher = LightGlueOpenCVMatcher(mock_logger, "lightglueopencv", extractor, config={})
        assert matcher.scoreThreshold == 0.1

    def test_custom_score_threshold(self, mock_logger):
        extractor = DISKOpenCV("disk", mock_logger, config={})
        matcher = LightGlueOpenCVMatcher(
            mock_logger, "lightglueopencv", extractor,
            config={'score_threshold': 0.5}
        )
        assert matcher.scoreThreshold == 0.5

    def test_default_mode_is_simple(self, mock_logger):
        extractor = DISKOpenCV("disk", mock_logger, config={})
        matcher = LightGlueOpenCVMatcher(mock_logger, "lightglueopencv", extractor, config={})
        assert matcher.mode == 'simple'

    def test_custom_mode_knn(self, mock_logger):
        extractor = DISKOpenCV("disk", mock_logger, config={})
        matcher = LightGlueOpenCVMatcher(
            mock_logger, "lightglueopencv", extractor,
            config={'mode': 'knn'}
        )
        assert matcher.mode == 'knn'

    def test_unknown_config_key_consumed(self, mock_logger):
        extractor = DISKOpenCV("disk", mock_logger, config={})
        matcher = LightGlueOpenCVMatcher(
            mock_logger, "lightglueopencv", extractor,
            config={'unknown_key': 123, 'score_threshold': 0.2}
        )
        assert matcher.scoreThreshold == 0.2


class TestLightGlueInference:
    def test_match_returns_dict_structure(self, mock_logger, tmp_features):
        extractor = DISKOpenCV("disk", mock_logger, config={})
        matcher = LightGlueOpenCVMatcher(mock_logger, "lightglueopencv", extractor, config={})
        result = matcher.match(tmp_features, tmp_features)

        assert isinstance(result, dict)
        assert 'matches' in result
        assert isinstance(result['matches'], tuple)

    def test_match_simple_mode_works(self, mock_logger, tmp_features):
        extractor = DISKOpenCV("disk", mock_logger, config={})
        matcher = LightGlueOpenCVMatcher(
            mock_logger, "lightglueopencv", extractor,
            config={'mode': 'simple'}
        )
        result = matcher.match(tmp_features, tmp_features)

        assert 'matches' in result
        for m in result['matches']:
            assert isinstance(m, cv.DMatch)

    def test_match_knn_mode_works(self, mock_logger, tmp_features):
        extractor = DISKOpenCV("disk", mock_logger, config={})
        matcher = LightGlueOpenCVMatcher(
            mock_logger, "lightglueopencv", extractor,
            config={'mode': 'knn'}
        )
        result = matcher.match(tmp_features, tmp_features)

        assert 'matches' in result
        for m_list in result['matches']:
            assert isinstance(m_list, (list, tuple))
            if len(m_list) > 0:
                assert isinstance(m_list[0], cv.DMatch)

    def test_match_returns_empty_on_none_descriptors(self, mock_logger, tmp_features):
        extractor = DISKOpenCV("disk", mock_logger, config={})
        matcher = LightGlueOpenCVMatcher(mock_logger, "lightglueopencv", extractor, config={})
        features_no_des = {**tmp_features, 'des': None}

        result = matcher.match(features_no_des, tmp_features)
        assert result == {'matches': ()}

    def test_match_returns_empty_on_none_keypoints(self, mock_logger, tmp_features):
        extractor = DISKOpenCV("disk", mock_logger, config={})
        matcher = LightGlueOpenCVMatcher(mock_logger, "lightglueopencv", extractor, config={})
        features_no_kp = {**tmp_features, 'kp': None}

        result = matcher.match(features_no_kp, tmp_features)
        assert result == {'matches': ()}


class TestLightGlueRobustness:
    def test_match_with_empty_keypoints(self, mock_logger, tmp_features):
        extractor = DISKOpenCV("disk", mock_logger, config={})
        matcher = LightGlueOpenCVMatcher(mock_logger, "lightglueopencv", extractor, config={})
        empty_features = {**tmp_features, 'kp': []}

        result = matcher.match(empty_features, tmp_features)
        assert isinstance(result, dict)
        assert 'matches' in result

    def test_match_with_different_image_sizes(self, mock_logger, load_img):
        extractor = DISKOpenCV("disk", mock_logger, config={})
        matcher = LightGlueOpenCVMatcher(mock_logger, "lightglueopencv", extractor, config={})

        img1 = load_img("box.png")
        img2 = load_img("box_in_scene.png")

        kp1 = [cv.KeyPoint(100, 100, 10)]
        kp2 = [cv.KeyPoint(150, 150, 10)]
        des1 = np.random.rand(1, 128).astype(np.float32)
        des2 = np.random.rand(1, 128).astype(np.float32)

        features1 = {'kp': kp1, 'des': des1, 'img_shape': img1.shape}
        features2 = {'kp': kp2, 'des': des2, 'img_shape': img2.shape}

        result = matcher.match(features1, features2)
        assert isinstance(result, dict)
        assert 'matches' in result
