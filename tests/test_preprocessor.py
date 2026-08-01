import pytest
import cv2 as cv
import numpy as np
import torch
from pathlib import Path
from unittest.mock import MagicMock
from logging import Logger

from src.preprocessor import Preprocessor
from src.algorithms import DNN_ALGORITHMS, OPENCV_ALGORITHMS


@pytest.fixture
def mock_logger():
    return MagicMock(spec=Logger)


@pytest.fixture
def preprocessor(mock_logger):
    return Preprocessor(logger=mock_logger, config=None)


@pytest.fixture
def preprocessor_cpu(mock_logger):
    return Preprocessor(logger=mock_logger, config={'device': 'cpu'})


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
def sample_opencv_features():
    kp = [cv.KeyPoint(x=10.0, y=20.0, size=8),
          cv.KeyPoint(x=30.0, y=40.0, size=8)]
    des = np.random.rand(2, 128).astype(np.float32)
    return {'kp': kp, 'des': des}


@pytest.fixture
def sample_neural_features():
    return {
        'keypoints': np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        'descriptors': np.random.rand(2, 256).astype(np.float32),
        'keypoint_scores': np.array([0.9, 0.8], dtype=np.float32)
    }


@pytest.fixture
def sample_neural_matches():
    return {
        'matches': torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        'scores': torch.tensor([0.9, 0.8])
    }


@pytest.fixture
def sample_opencv_matches():
    matches = [cv.DMatch(_queryIdx=0, _trainIdx=1, _distance=0.5),
               cv.DMatch(_queryIdx=1, _trainIdx=0, _distance=0.3)]
    return {'matches': matches, 'scores': None}


class TestPreProcessorInit:
    def test_default_config(self, mock_logger):
        pp = Preprocessor(logger=mock_logger)
        assert pp._device == 'cpu'

    def test_none_config(self, mock_logger):
        pp = Preprocessor(logger=mock_logger, config=None)
        assert pp._device == 'cpu'

    def test_custom_device(self, mock_logger):
        pp = Preprocessor(logger=mock_logger, config={'device': 'cpu'})
        assert pp._device == 'cpu'

    def test_empty_config(self, mock_logger):
        pp = Preprocessor(logger=mock_logger, config={})
        assert pp._device == 'cpu'

    def test_logger_assigned(self, mock_logger):
        pp = Preprocessor(logger=mock_logger)
        assert pp._logger is mock_logger

    def test_converters_created(self, preprocessor):
        assert preprocessor._image_converter is not None
        assert preprocessor._features_converter is not None
        assert preprocessor._matches_converter is not None


class TestPreProcessorGetFormat:
    def test_neural_algo_returns_tensor(self, preprocessor):
        for algo in DNN_ALGORITHMS:
            assert preprocessor._get_format(algo) == 'tensor'

    def test_opencv_algo_returns_opencv(self, preprocessor):
        for algo in OPENCV_ALGORITHMS:
            assert preprocessor._get_format(algo) == 'opencv'

    def test_explicit_tensor_format(self, preprocessor):
        assert preprocessor._get_format('tensor') == 'tensor'

    def test_explicit_opencv_format(self, preprocessor):
        assert preprocessor._get_format('opencv') == 'opencv'

    def test_unknown_algo_raises(self, preprocessor):
        with pytest.raises(ValueError, match="Unknown algorithm or format"):
            preprocessor._get_format('unknown_algo')


class TestPreProcessorLogConversion:
    def test_logs_when_formats_differ(self, preprocessor, mock_logger):
        preprocessor._log_conversion('image', 'sift', 'superpoint')
        mock_logger.info.assert_called_once()

    def test_no_log_when_formats_same(self, preprocessor, mock_logger):
        preprocessor._log_conversion('image', 'sift', 'orb')
        mock_logger.info.assert_not_called()

    def test_returns_correct_formats(self, preprocessor):
        from_fmt, to_fmt = preprocessor._log_conversion('image', 'sift', 'superpoint')
        assert from_fmt == 'opencv'
        assert to_fmt == 'tensor'

    def test_log_message_contains_data_type(self, preprocessor, mock_logger):
        preprocessor._log_conversion('features', 'sift', 'superpoint')
        call_args = mock_logger.info.call_args[0][0]
        assert 'features' in call_args

    def test_log_message_contains_device(self, preprocessor, mock_logger):
        preprocessor._log_conversion('image', 'sift', 'superpoint')
        call_args = mock_logger.info.call_args[0][0]
        assert preprocessor._device in call_args


class TestPreProcessorPrepareImage:
    def test_opencv_to_opencv_no_conversion(self, preprocessor, load_img):
        img = load_img("box.png")
        result = preprocessor.prepare_image(img, from_algo='sift', to_algo='orb')
        assert isinstance(result, np.ndarray)
        assert result is img

    def test_opencv_to_tensor(self, preprocessor, load_img):
        img = load_img("box.png")
        result = preprocessor.prepare_image(img, from_algo='opencv', to_algo='superpoint_lightglue')
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 3
        assert result.shape[0] == 3

    def test_tensor_to_opencv(self, preprocessor):
        tensor = torch.rand(3, 100, 100)
        result = preprocessor.prepare_image(tensor, from_algo='superpoint_lightglue', to_algo='opencv')
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3

    def test_same_format_no_log(self, preprocessor, mock_logger, load_img):
        img = load_img("box.png")
        preprocessor.prepare_image(img, from_algo='sift', to_algo='orb')
        mock_logger.info.assert_not_called()

    def test_different_format_logs(self, preprocessor, mock_logger, load_img):
        img = load_img("box.png")
        preprocessor.prepare_image(img, from_algo='sift', to_algo='superpoint')
        mock_logger.info.assert_called_once()

    def test_explicit_formats(self, preprocessor, load_img):
        img = load_img("box.png")
        result = preprocessor.prepare_image(img, from_algo='opencv', to_algo='tensor')
        assert isinstance(result, torch.Tensor)


class TestPreProcessorPrepareFeatures:
    def test_opencv_to_opencv_no_conversion(self, preprocessor, sample_opencv_features):
        result = preprocessor.prepare_features(sample_opencv_features,
                                               from_algo='sift', to_algo='orb')
        assert result is sample_opencv_features

    def test_opencv_to_neural(self, preprocessor, sample_opencv_features):
        result = preprocessor.prepare_features(sample_opencv_features,
                                               from_algo='sift', to_algo='superpoint')
        assert 'keypoints' in result
        assert 'descriptors' in result
        assert isinstance(result['keypoints'], torch.Tensor)

    def test_neural_to_neural_no_conversion(self, preprocessor, sample_neural_features):
        result = preprocessor.prepare_features(sample_neural_features,
                                               from_algo='superpoint',
                                               to_algo='superpoint_lightglue')
        assert result is sample_neural_features

    def test_no_log_same_format(self, preprocessor, mock_logger, sample_opencv_features):
        preprocessor.prepare_features(sample_opencv_features,
                                      from_algo='sift', to_algo='orb')
        mock_logger.info.assert_not_called()


class TestPreProcessorPrepareMatches:
    def test_neural_to_opencv(self, preprocessor, sample_neural_matches):
        result = preprocessor.prepare_matches(sample_neural_matches,
                                              from_algo='lightglue', to_algo='sift')
        assert 'matches' in result
        assert isinstance(result['matches'], list)
        if result['matches']:
            assert isinstance(result['matches'][0], cv.DMatch)

    def test_opencv_to_neural(self, preprocessor, sample_opencv_matches):
        result = preprocessor.prepare_matches(sample_opencv_matches,
                                              from_algo='bf', to_algo='superpoint')
        assert 'matches' in result
        assert isinstance(result['matches'], torch.Tensor)

    def test_opencv_to_opencv_no_conversion(self, preprocessor, sample_opencv_matches):
        result = preprocessor.prepare_matches(sample_opencv_matches,
                                              from_algo='bf', to_algo='flann')
        assert result is sample_opencv_matches

    def test_empty_neural_matches(self, preprocessor):
        empty_matches = {'matches': torch.zeros((0, 2), dtype=torch.long), 'scores': None}
        result = preprocessor.prepare_matches(empty_matches,
                                              from_algo='lightglue', to_algo='sift')
        assert result['matches'] == []

    def test_match_indices_preserved(self, preprocessor, sample_neural_matches):
        result = preprocessor.prepare_matches(sample_neural_matches,
                                              from_algo='lightglue', to_algo='sift')
        matches = result['matches']
        assert matches[0].queryIdx == 0
        assert matches[0].trainIdx == 1

    def test_logs_conversion(self, preprocessor, mock_logger, sample_neural_matches):
        preprocessor.prepare_matches(sample_neural_matches,
                                     from_algo='lightglue', to_algo='sift')
        mock_logger.info.assert_called_once()

    def test_no_log_same_format(self, preprocessor, mock_logger, sample_opencv_matches):
        preprocessor.prepare_matches(sample_opencv_matches,
                                     from_algo='bf', to_algo='flann')
        mock_logger.info.assert_not_called()
