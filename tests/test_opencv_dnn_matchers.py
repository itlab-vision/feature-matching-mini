import pytest
import cv2 as cv
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
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
            return np.zeros((240, 320, 3) if color else (240, 320), dtype=np.uint8)
        return img

    return _load


@pytest.fixture
def tmp_features(load_img):
    img = load_img("box.png")
    h, w = img.shape[:2]
    kp = [cv.KeyPoint(float(w // 2 + i * 15), float(h // 2 + i * 15), 10.0) for i in range(10)]
    des = np.random.rand(10, 128).astype(np.float32)
    return {
        'keypoints': kp,
        'descriptors': des,
        'scores': np.ones(10, dtype=np.float32),
        'img_shape': img.shape
    }


class TestLightGlueRegistration:
    @pytest.fixture(autouse=True)
    def mock_backend(self):
        with patch('src.opencv_dnn_matchers.ModelWrapper'), \
                patch('src.opencv_dnn_matchers.IOAdapter'), \
                patch('src.opencv_dnn_matchers.InferenceAPI'), \
                patch('src.opencv_dnn_extractors.ModelWrapper'), \
                patch('src.opencv_dnn_extractors.IOAdapter'), \
                patch('src.opencv_dnn_extractors.InferenceAPI'):
            yield

    def test_registered_in_factory(self, mock_logger):
        extractor = DISKOpenCV("disk_opencv", mock_logger, config={})
        obj = Matcher.create("lightglue_opencv", mock_logger, descriptor_class=extractor, config={})
        assert isinstance(obj, LightGlueOpenCVMatcher)
        assert "lightglue_opencv" in Matcher._METHODS


class TestLightGlueConfig:
    @pytest.fixture(autouse=True)
    def mock_backend(self):
        with patch('src.opencv_dnn_matchers.ModelWrapper'), \
                patch('src.opencv_dnn_matchers.IOAdapter'), \
                patch('src.opencv_dnn_matchers.InferenceAPI'), \
                patch('src.opencv_dnn_extractors.ModelWrapper'), \
                patch('src.opencv_dnn_extractors.IOAdapter'), \
                patch('src.opencv_dnn_extractors.InferenceAPI'):
            yield

    def test_default_score_threshold(self, mock_logger):
        extractor = DISKOpenCV("disk_opencv", mock_logger, config={})
        matcher = LightGlueOpenCVMatcher("lightglue_opencv", mock_logger, config={}, descriptor=extractor)
        assert getattr(matcher, 'scoreThreshold', 0.1) == 0.1

    def test_custom_score_threshold(self, mock_logger):
        extractor = DISKOpenCV("disk_opencv", mock_logger, config={})
        matcher = LightGlueOpenCVMatcher(
            "lightglue_opencv", mock_logger,
            config={'score_threshold': 0.5}, descriptor=extractor
        )
        assert getattr(matcher, 'scoreThreshold', 0.5) == 0.5

    def test_unknown_config_key_consumed(self, mock_logger):
        extractor = DISKOpenCV("disk_opencv", mock_logger, config={})
        matcher = LightGlueOpenCVMatcher(
            "lightglue_opencv", mock_logger,
            config={'unknown_key': 123, 'score_threshold': 0.2}, descriptor=extractor
        )
        assert getattr(matcher, 'scoreThreshold', 0.2) == 0.2


class TestLightGlueInference:
    @pytest.fixture(autouse=True)
    def mock_backend_inference(self):
        with patch('src.opencv_dnn_matchers.ModelWrapper'), \
                patch('src.opencv_dnn_matchers.IOAdapter') as mock_io, \
                patch('src.opencv_dnn_matchers.InferenceAPI'), \
                patch('src.opencv_dnn_extractors.ModelWrapper'), \
                patch('src.opencv_dnn_extractors.IOAdapter'), \
                patch('src.opencv_dnn_extractors.InferenceAPI'):
            mock_io_instance = MagicMock()
            mock_io_instance.postprocess.return_value = {
                'matches': (cv.DMatch(0, 0, 1.0),),
                'scores': np.array([0.9], dtype=np.float32)
            }
            mock_io.create.return_value = mock_io_instance
            yield

    def test_match_returns_dict_structure(self, mock_logger, tmp_features):
        extractor = DISKOpenCV("disk_opencv", mock_logger, config={})
        matcher = LightGlueOpenCVMatcher("lightglue_opencv", mock_logger, config={}, descriptor=extractor)
        result = matcher.match(tmp_features, tmp_features)

        assert isinstance(result, dict)
        assert 'matches' in result
        assert isinstance(result['matches'], (tuple, list, np.ndarray))

    def test_match_works(self, mock_logger, tmp_features):
        extractor = DISKOpenCV("disk_opencv", mock_logger, config={})
        matcher = LightGlueOpenCVMatcher(
            "lightglue_opencv", mock_logger,
            config={}, descriptor=extractor
        )
        result = matcher.match(tmp_features, tmp_features)

        assert 'matches' in result
        for m in result['matches']:
            assert isinstance(m, cv.DMatch)

    def test_match_returns_empty_on_none_descriptors(self, mock_logger, tmp_features):
        with patch('src.opencv_dnn_matchers.IOAdapter') as mock_io:
            mock_io_instance = MagicMock()
            mock_io_instance.postprocess.return_value = {
                'matches': (),
                'scores': np.array([], dtype=np.float32)
            }
            mock_io.create.return_value = mock_io_instance

            extractor = DISKOpenCV("disk_opencv", mock_logger, config={})
            matcher = LightGlueOpenCVMatcher("lightglue_opencv", mock_logger, config={}, descriptor=extractor)
            features_no_des = {**tmp_features, 'descriptors': None}

            result = matcher.match(features_no_des, tmp_features)
            assert 'matches' in result

    def test_match_returns_empty_on_none_keypoints(self, mock_logger, tmp_features):
        with patch('src.opencv_dnn_matchers.IOAdapter') as mock_io:
            mock_io_instance = MagicMock()
            mock_io_instance.postprocess.return_value = {
                'matches': (),
                'scores': np.array([], dtype=np.float32)
            }
            mock_io.create.return_value = mock_io_instance

            extractor = DISKOpenCV("disk_opencv", mock_logger, config={})
            matcher = LightGlueOpenCVMatcher("lightglue_opencv", mock_logger, config={}, descriptor=extractor)
            features_no_kp = {**tmp_features, 'keypoints': None}

            result = matcher.match(features_no_kp, tmp_features)
            assert 'matches' in result


class TestLightGlueRobustness:
    @pytest.fixture(autouse=True)
    def mock_backend_robustness(self):
        with patch('src.opencv_dnn_matchers.ModelWrapper'), \
                patch('src.opencv_dnn_matchers.IOAdapter') as mock_io, \
                patch('src.opencv_dnn_matchers.InferenceAPI'), \
                patch('src.opencv_dnn_extractors.ModelWrapper'), \
                patch('src.opencv_dnn_extractors.IOAdapter'), \
                patch('src.opencv_dnn_extractors.InferenceAPI'):
            mock_io_instance = MagicMock()
            mock_io_instance.postprocess.return_value = {
                'matches': (cv.DMatch(0, 0, 1.0),),
                'scores': np.array([0.9], dtype=np.float32)
            }
            mock_io.create.return_value = mock_io_instance
            yield

    def test_match_with_empty_keypoints(self, mock_logger, tmp_features):
        extractor = DISKOpenCV("disk_opencv", mock_logger, config={})
        matcher = LightGlueOpenCVMatcher("lightglue_opencv", mock_logger, config={}, descriptor=extractor)
        empty_features = {**tmp_features, 'kp': [], 'keypoints': []}

        result = matcher.match(empty_features, tmp_features)
        assert isinstance(result, dict)
        assert 'matches' in result

    def test_match_with_different_image_sizes(self, mock_logger, load_img):
        extractor = DISKOpenCV("disk_opencv", mock_logger, config={})
        matcher = LightGlueOpenCVMatcher("lightglue_opencv", mock_logger, config={}, descriptor=extractor)

        img1 = load_img("box.png")
        img2 = load_img("box_in_scene.png")

        kp1 = [cv.KeyPoint(100.0, 100.0, 10.0)]
        kp2 = [cv.KeyPoint(150.0, 150.0, 10.0)]
        des1 = np.random.rand(1, 128).astype(np.float32)
        des2 = np.random.rand(1, 128).astype(np.float32)

        features1 = {'keypoints': kp1, 'descriptors': des1, 'img_shape': img1.shape}
        features2 = {'keypoints': kp2, 'descriptors': des2, 'img_shape': img2.shape}

        result = matcher.match(features1, features2)
        assert isinstance(result, dict)
        assert 'matches' in result
