import pytest
import cv2 as cv
import numpy as np
import torch

from feature_matching.converter import (Converter, ImageConverter, FeaturesConverter, MatchesConverter)


@pytest.fixture
def sample_numpy_image():
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_tensor_image():
    return torch.rand(3, 100, 100)


@pytest.fixture
def sample_numpy_grayscale():
    return np.random.randint(0, 255, (100, 100), dtype=np.uint8)


@pytest.fixture
def sample_tensor_grayscale():
    return torch.rand(1, 100, 100)


@pytest.fixture
def sample_features_cv():
    kp = [cv.KeyPoint(10, 20, 5), cv.KeyPoint(30, 40, 5)]
    des = np.random.rand(2, 128).astype(np.float32)
    return {'kp': kp, 'des': des}


@pytest.fixture
def sample_features_tensor():
    keypoints = torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32)
    descriptors = torch.randn(2, 128, dtype=torch.float32)
    return {'keypoints': keypoints, 'descriptors': descriptors}


@pytest.fixture
def sample_matches_cv():
    dmatches = []
    for i in range(5):
        dmatch = cv.DMatch()
        dmatch.queryIdx = i
        dmatch.trainIdx = i * 2
        dmatches.append(dmatch)
    return {'matches': dmatches}


@pytest.fixture
def sample_matches_tensor():
    matches = torch.tensor([[0, 0], [1, 2], [2, 4], [3, 6], [4, 8]], dtype=torch.int64)
    return {'matches': matches}


class TestConverterRegistry:
    def test_registration_completeness(self):
        expected_converters = {"image", "features", "matches"}
        assert expected_converters.issubset(Converter._CONVERTERS.keys())

    def test_factory_creation_image(self):
        converter = Converter.create("image")
        assert isinstance(converter, ImageConverter)
        assert isinstance(converter, Converter)

    def test_factory_creation_features(self):
        converter = Converter.create("features")
        assert isinstance(converter, FeaturesConverter)
        assert isinstance(converter, Converter)

    def test_factory_creation_matches(self):
        converter = Converter.create("matches")
        assert isinstance(converter, MatchesConverter)
        assert isinstance(converter, Converter)

    def test_factory_unknown_type(self):
        with pytest.raises(ValueError, match="Converter 'unknown' not found"):
            Converter.create("unknown")


class TestImageConverter:
    def test_standard_tensor(self, sample_tensor_image):
        converter = Converter.create('image')
        result = converter.convert(sample_tensor_image, 'tensor', 'opencv')

        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_grayscale_tensor(self, sample_tensor_grayscale):
        converter = Converter.create('image')
        result = converter.convert(sample_tensor_grayscale, 'tensor', 'opencv')

        assert result.shape == (100, 100, 1)
        assert result.dtype == np.uint8

    def test_2d_tensor(self):
        tensor = torch.rand(100, 100)
        converter = Converter.create('image')
        result = converter.convert(tensor, 'tensor', 'opencv')

        assert result.shape == (100, 100, 1)

    def test_normalization_tensor_to_cv(self):
        tensor = torch.tensor([0.0, 0.5, 1.0]).reshape(3, 1, 1)
        converter = Converter.create('image')
        result = converter.convert(tensor, 'tensor', 'opencv')

        assert result[0, 0, 0] == 255
        assert result[0, 0, 1] == 127
        assert result[0, 0, 2] == 0

    def test_standard_image(self, sample_numpy_image):
        converter = Converter.create('image')
        result = converter.convert(sample_numpy_image, 'opencv', 'tensor')

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 100, 100)
        assert result.dtype == torch.float32
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_grayscale_image(self):
        gray_img = np.random.randint(0, 255, (100, 100, 1), dtype=np.uint8)
        converter = Converter.create('image')
        result = converter.convert(gray_img, 'opencv', 'tensor')

        assert result.shape == (1, 100, 100)
        assert result.dtype == torch.float32

    def test_normalization_cv_to_tensor(self):
        img = np.array([[[255, 0, 0]]], dtype=np.uint8)
        converter = Converter.create('image')
        result = converter.convert(img, 'opencv', 'tensor')

        assert result[0, 0, 0] == 0.0
        assert result[1, 0, 0] == 0.0
        assert result[2, 0, 0] == 1.0


class TestFeaturesConverter:
    def test_opencv_to_tensor(self, sample_features_cv):
        converter = FeaturesConverter()
        result = converter.convert(sample_features_cv, 'opencv', 'tensor')

        assert isinstance(result, dict)
        assert 'keypoints' in result
        assert 'descriptors' in result
        assert isinstance(result['keypoints'], torch.Tensor)
        assert isinstance(result['descriptors'], torch.Tensor)
        assert result['keypoints'].shape == (2, 2)
        assert result['descriptors'].shape == (2, 128)

    def test_tensor_to_opencv(self, sample_features_tensor):
        converter = FeaturesConverter()
        result = converter.convert(sample_features_tensor, 'tensor', 'opencv')

        assert isinstance(result, dict)
        assert 'kp' in result
        assert 'des' in result
        assert isinstance(result['kp'], tuple)
        assert isinstance(result['des'], np.ndarray)
        assert len(result['kp']) == 2
        assert result['des'].shape == (2, 128)


class TestMatchesConverter:
    def test_opencv_to_tensor(self, sample_matches_cv):
        converter = MatchesConverter()
        result = converter.convert(sample_matches_cv, 'opencv', 'tensor')

        assert isinstance(result, dict)
        assert 'matches' in result
        assert isinstance(result['matches'], torch.Tensor)
        assert result['matches'].shape == (5, 2)

    def test_tensor_to_opencv(self, sample_matches_tensor):
        converter = MatchesConverter()
        result = converter.convert(sample_matches_tensor, 'tensor', 'opencv')

        assert isinstance(result, dict)
        assert 'matches' in result
        assert isinstance(result['matches'], list)
        assert len(result['matches']) == 5
        assert isinstance(result['matches'][0], cv.DMatch)
