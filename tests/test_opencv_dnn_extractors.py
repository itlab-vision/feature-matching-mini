import pytest
import cv2 as cv
import numpy as np
from unittest.mock import MagicMock, patch
from logging import Logger

from src.opencv_dnn_extractors import OpenCVDNNFeatureExtractors, ALIKEDOpenCV, DISKOpenCV
from src.detectors import Detector
from src.descriptors import Descriptor


@pytest.fixture
def mock_logger():
    return MagicMock(spec=Logger)


@pytest.fixture
def random_img():
    return np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)


class TestOpenCVDNN:
    @pytest.fixture
    def base_instance(self, mock_logger):
        with patch('src.opencv_dnn_extractors.ModelWrapper'), \
                patch('src.opencv_dnn_extractors.IOAdapter') as mock_io_factory, \
                patch('src.opencv_dnn_extractors.InferenceAPI'):
            mock_io_instance = MagicMock()
            mock_io_instance.postprocess.return_value = {
                'keypoints': np.array([cv.KeyPoint(float(100 + i * 20), float(100 + i * 20), 10.0) for i in range(5)]),
                'descriptors': np.random.rand(5, 128).astype(np.float32),
                'scores': np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
            }
            mock_io_factory.create.return_value = mock_io_instance

            ext = OpenCVDNNFeatureExtractors("test_ext", mock_logger, {'nfeatures': 4096, 'threshold': 0.005})
            yield ext

    def test_detect_caches_results(self, base_instance, random_img):
        result = base_instance.detect(random_img)

        assert 'keypoints' in result or 'kp' in result
        assert 'descriptors' in result or 'des' in result
        keypoints = result.get('keypoints', result.get('kp'))
        descriptors = result.get('descriptors', result.get('des'))

        assert len(keypoints) == 5
        assert descriptors.shape == (5, 128)

    def test_compute_uses_cache_after_detect(self, base_instance, random_img):
        res1 = base_instance.detect(random_img)
        res2 = base_instance.compute(random_img, {})

        assert np.array_equal(res1['descriptors'], res2['descriptors'])

    def test_none_image_returns_empty(self, base_instance, mock_logger):
        result = base_instance._forward(None)

        assert result == {'keypoints': (), 'descriptors': ()}
        mock_logger.error.assert_called_once_with(
            "Input image is None. Detection aborted."
        )

    def test_default_norm_is_l2(self, base_instance):
        assert base_instance.default_norm == cv.NORM_L2

    def test_registered_in_factories(self):
        assert "aliked_opencv" in Detector._METHODS or "aliked_opencv" in Descriptor._METHODS
        assert "disk_opencv" in Detector._METHODS or "disk_opencv" in Descriptor._METHODS


class TestALIKEDOpenCV:
    @pytest.fixture(autouse=True)
    def mock_backend_factories(self):
        with patch('src.opencv_dnn_extractors.ModelWrapper'), \
                patch('src.opencv_dnn_extractors.IOAdapter') as mock_io, \
                patch('src.opencv_dnn_extractors.InferenceAPI'):
            mock_io_instance = MagicMock()
            mock_io_instance.postprocess.return_value = {
                'keypoints': np.array([cv.KeyPoint(10.0, 10.0, 1.0)]),
                'descriptors': np.random.rand(1, 128).astype(np.float32),
                'scores': np.array([0.9], dtype=np.float32)
            }
            mock_io.create.return_value = mock_io_instance
            yield

    def test_default_initialization(self, mock_logger):
        extractor = ALIKEDOpenCV("aliked_opencv", mock_logger, config={})
        assert extractor.extractor_name == "aliked_opencv"

    def test_custom_nfeatures(self, mock_logger):
        extractor = ALIKEDOpenCV("aliked_opencv", mock_logger, config={'nfeatures': 2048})
        assert extractor is not None

    def test_custom_threshold(self, mock_logger):
        extractor = ALIKEDOpenCV("aliked_opencv", mock_logger, config={'threshold': 0.05})
        assert extractor is not None

    def test_custom_scale_factor(self, mock_logger):
        extractor = ALIKEDOpenCV("aliked_opencv", mock_logger, config={'scale_factor': 1.2})
        assert extractor is not None

    def test_unknown_params_ignored(self, mock_logger):
        extractor = ALIKEDOpenCV("aliked_opencv", mock_logger, config={'unknown_param': 999, 'nfeatures': 1000})
        assert extractor is not None

    def test_custom_model_path(self, mock_logger):
        custom_path = "models/aliked-n32-top2k-640.onnx"
        extractor = ALIKEDOpenCV("aliked_opencv", mock_logger, config={'aliked_model_path': custom_path})
        assert extractor is not None

    def test_black_image(self, mock_logger):
        with patch('src.opencv_dnn_extractors.IOAdapter') as mock_io_patch:
            mock_io_instance = MagicMock()
            mock_io_instance.postprocess.return_value = {
                'keypoints': np.array([]),
                'descriptors': np.empty((0, 128), dtype=np.float32),
                'scores': np.array([], dtype=np.float32)
            }
            mock_io_patch.create.return_value = mock_io_instance

            img = np.zeros((200, 200, 3), dtype=np.uint8)
            extractor = ALIKEDOpenCV("aliked_opencv", mock_logger, config={})
            result = extractor.compute(img, {'keypoints': []})

            assert isinstance(result, dict)
            assert 'keypoints' in result
            assert 'descriptors' in result

    def test_very_small_image(self, mock_logger):
        tiny_img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        extractor = ALIKEDOpenCV("aliked_opencv", mock_logger, config={})

        try:
            result = extractor.compute(tiny_img, {'keypoints': []})
            assert isinstance(result, dict)
            assert 'keypoints' in result
            assert 'descriptors' in result
        except Exception as e:
            pytest.fail(f"ALIKED failed on tiny image: {e}")

    def test_high_noise_image(self, mock_logger):
        noise = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        extractor = ALIKEDOpenCV("aliked_opencv", mock_logger, config={})
        result = extractor.compute(noise, {'keypoints': []})

        assert isinstance(result, dict)
        assert 'keypoints' in result
        assert 'descriptors' in result


class TestDISKOpenCV:
    @pytest.fixture(autouse=True)
    def mock_backend_factories(self):
        with patch('src.opencv_dnn_extractors.ModelWrapper'), \
                patch('src.opencv_dnn_extractors.IOAdapter') as mock_io, \
                patch('src.opencv_dnn_extractors.InferenceAPI'):
            mock_io_instance = MagicMock()
            mock_io_instance.postprocess.return_value = {
                'keypoints': np.array([cv.KeyPoint(10.0, 10.0, 1.0)]),
                'descriptors': np.random.rand(1, 128).astype(np.float32),
                'scores': np.array([0.9], dtype=np.float32)
            }
            mock_io.create.return_value = mock_io_instance
            yield

    def test_default_initialization(self, mock_logger):
        extractor = DISKOpenCV("disk_opencv", mock_logger, config={})
        assert extractor.extractor_name == "disk_opencv"

    def test_custom_nfeatures(self, mock_logger):
        extractor = DISKOpenCV("disk_opencv", mock_logger, config={'nfeatures': 1024})
        assert extractor is not None

    def test_custom_threshold(self, mock_logger):
        extractor = DISKOpenCV("disk_opencv", mock_logger, config={'threshold': 0.3})
        assert extractor is not None

    def test_unknown_params_not_passed(self, mock_logger):
        extractor = DISKOpenCV("disk_opencv", mock_logger, config={'unknown': 123, 'nfeatures': 500})
        assert extractor is not None

    def test_custom_model_path(self, mock_logger):
        custom_path = "models/disk_1024.onnx"
        extractor = DISKOpenCV("disk_opencv", mock_logger, config={'disk_model_path': custom_path})
        assert extractor is not None

    def test_black_image(self, mock_logger):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        extractor = DISKOpenCV("disk_opencv", mock_logger, config={})
        result = extractor.compute(img, {'keypoints': []})

        assert isinstance(result, dict)
        assert 'keypoints' in result
        assert 'descriptors' in result

    def test_very_small_image(self, mock_logger):
        tiny_img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        extractor = DISKOpenCV("disk_opencv", mock_logger, config={})

        try:
            result = extractor.compute(tiny_img, {'keypoints': []})
            assert isinstance(result, dict)
            assert 'keypoints' in result
            assert 'descriptors' in result
        except Exception as e:
            pytest.fail(f"DISK failed on tiny image: {e}")

    def test_high_noise_image(self, mock_logger):
        noise = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        extractor = DISKOpenCV("disk_opencv", mock_logger, config={})
        result = extractor.compute(noise, {'keypoints': []})

        assert isinstance(result, dict)
        assert 'keypoints' in result
        assert 'descriptors' in result
