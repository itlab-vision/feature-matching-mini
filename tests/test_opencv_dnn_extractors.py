import pytest
import cv2 as cv
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
from logging import Logger

from feature_matching.opencv_dnn_extractors import OpenCVDNNFeatureExtractors, ALIKEDOpenCV, DISKOpenCV
from feature_matching.detectors import Detector
from feature_matching.descriptors import Descriptor


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


class TestOpenCVDNN:
    @pytest.fixture
    def mock_extractor(self):
        extractor = MagicMock()
        kp = [cv.KeyPoint(100 + i * 20, 100 + i * 20, 10) for i in range(5)]
        des = np.random.rand(5, 128).astype(np.float32)
        extractor.detectAndCompute.return_value = (kp, des)
        return extractor

    @pytest.fixture
    def base_instance(self, mock_logger, mock_extractor):
        return OpenCVDNNFeatureExtractors("test_ext", mock_logger, mock_extractor)

    def test_detect_caches_results(self, base_instance, random_img, mock_extractor):
        result = base_instance.detect(random_img)

        assert 'kp' in result
        assert 'des' in result
        assert len(result['kp']) == 5
        assert result['des'].shape == (5, 128)
        assert mock_extractor.detectAndCompute.call_count == 1

    def test_compute_uses_cache_after_detect(self, base_instance, random_img, mock_extractor):
        base_instance.detect(random_img)
        first_call_count = mock_extractor.detectAndCompute.call_count

        base_instance.compute(random_img, {})
        second_call_count = mock_extractor.detectAndCompute.call_count

        assert first_call_count == second_call_count

    def test_none_image_returns_empty(self, base_instance, mock_logger):
        result = base_instance._forward(None)

        assert result == {'kp': (), 'des': ()}
        mock_logger.error.assert_called_once_with(
            "Input image is None. Detection aborted."
        )

    def test_default_norm_is_l2(self, base_instance):
        assert base_instance.default_norm == cv.NORM_L2

    def test_registered_in_factories(self):
        assert "OpenCVDNNFeatureExtractors" not in Detector._METHODS
        assert "OpenCVDNNFeatureExtractors" not in Descriptor._METHODS
        assert "alikedopencv" in Detector._METHODS or "aliked" in Detector._METHODS
        assert "diskopencv" in Detector._METHODS or "disk" in Detector._METHODS


class TestALIKEDOpenCV:
    def test_default_initialization(self, mock_logger):
        extractor = ALIKEDOpenCV("aliked", mock_logger, config={})

        assert extractor.extractor is not None
        assert extractor.extractor_name == "aliked"

    def test_custom_nfeatures(self, mock_logger):
        extractor = ALIKEDOpenCV(
            "aliked",
            mock_logger,
            config={'nfeatures': 2048}
        )

        assert extractor.extractor is not None

    def test_custom_threshold(self, mock_logger):
        extractor = ALIKEDOpenCV(
            "aliked",
            mock_logger,
            config={'threshold': 0.05}
        )
        assert extractor.extractor is not None

    def test_custom_scale_factor(self, mock_logger):
        extractor = ALIKEDOpenCV(
            "aliked",
            mock_logger,
            config={'scale_factor': 1.2}
        )
        assert extractor.extractor is not None

    def test_unknown_params_ignored(self, mock_logger):
        extractor = ALIKEDOpenCV(
            "aliked",
            mock_logger,
            config={'unknown_param': 999, 'nfeatures': 1000}
        )
        assert extractor.extractor is not None

    def test_custom_model_path(self, mock_logger):
        custom_path = "models/aliked-n32-top2k-640.onnx"
        extractor = ALIKEDOpenCV(
            "aliked",
            mock_logger,
            config={'aliked_model_path': custom_path}
        )
        assert extractor.extractor is not None

    def test_black_image(self, mock_logger):
        OpenCVDNNFeatureExtractors._is_extracted = False
        OpenCVDNNFeatureExtractors._extracted_data = {}

        img = np.zeros((200, 200), dtype=np.uint8)
        extractor = ALIKEDOpenCV("aliked", mock_logger, config={})
        result = extractor.compute(img, {'kp': []})

        assert isinstance(result, dict)
        assert 'kp' in result
        assert 'des' in result

    def test_very_small_image(self, mock_logger):
        OpenCVDNNFeatureExtractors._is_extracted = False
        OpenCVDNNFeatureExtractors._extracted_data = {}

        tiny_img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        extractor = ALIKEDOpenCV("aliked", mock_logger, config={})

        try:
            result = extractor.compute(tiny_img, {'kp': []})
            assert isinstance(result, dict)
            assert 'kp' in result
        except Exception as e:
            pytest.fail(f"ALIKED failed on tiny image: {e}")

    def test_high_noise_image(self, mock_logger):
        OpenCVDNNFeatureExtractors._is_extracted = False
        OpenCVDNNFeatureExtractors._extracted_data = {}

        noise = np.random.randint(0, 2, (200, 200), dtype=np.uint8) * 255
        extractor = ALIKEDOpenCV("aliked", mock_logger, config={})
        result = extractor.compute(noise, {'kp': []})

        assert isinstance(result, dict)
        assert 'des' in result


class TestDISKOpenCV:
    def test_default_initialization(self, mock_logger):
        extractor = DISKOpenCV("disk", mock_logger, config={})

        assert extractor.extractor is not None
        assert extractor.extractor_name == "disk"

    def test_custom_nfeatures(self, mock_logger):
        extractor = DISKOpenCV(
            "disk",
            mock_logger,
            config={'nfeatures': 1024}
        )
        assert extractor.extractor is not None

    def test_custom_threshold(self, mock_logger):
        extractor = DISKOpenCV(
            "disk",
            mock_logger,
            config={'threshold': 0.3}
        )
        assert extractor.extractor is not None

    def test_unknown_params_not_passed(self, mock_logger):
        extractor = DISKOpenCV(
            "disk",
            mock_logger,
            config={'unknown': 123, 'nfeatures': 500}
        )
        assert extractor.extractor is not None

    def test_custom_model_path(self, mock_logger):
        custom_path = "models/disk_1024.onnx"
        extractor = DISKOpenCV(
            "disk",
            mock_logger,
            config={'disk_model_path': custom_path}
        )
        assert extractor.extractor is not None

    def test_black_image(self, mock_logger):
        OpenCVDNNFeatureExtractors._is_extracted = False
        OpenCVDNNFeatureExtractors._extracted_data = {}

        img = np.zeros((200, 200), dtype=np.uint8)
        extractor = DISKOpenCV("disk", mock_logger, config={})
        result = extractor.compute(img, {'kp': []})

        assert isinstance(result, dict)
        assert 'kp' in result
        assert 'des' in result

    def test_very_small_image(self, mock_logger):
        OpenCVDNNFeatureExtractors._is_extracted = False
        OpenCVDNNFeatureExtractors._extracted_data = {}

        tiny_img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        extractor = DISKOpenCV("disk", mock_logger, config={})

        try:
            result = extractor.compute(tiny_img, {'kp': []})
            assert isinstance(result, dict)
            assert 'kp' in result
        except Exception as e:
            pytest.fail(f"DISK failed on tiny image: {e}")

    def test_high_noise_image(self, mock_logger):
        OpenCVDNNFeatureExtractors._is_extracted = False
        OpenCVDNNFeatureExtractors._extracted_data = {}

        noise = np.random.randint(0, 2, (200, 200), dtype=np.uint8) * 255
        extractor = DISKOpenCV("disk", mock_logger, config={})
        result = extractor.compute(noise, {'kp': []})

        assert isinstance(result, dict)
        assert 'des' in result
