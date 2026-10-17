
import cv2 as cv
import numpy as np
import pytest
import torch
from unittest.mock import patch

from src.backend.torch.utils import (is_tensor, empty, get_device, numpy_to_tensor, tensor_to_numpy,
                                     to_numpy_bgr, image_to_tensor, image_to_cv, features_to_tensor,
                                     features_to_cv, matches_to_tensor, matches_to_cv)


@pytest.fixture
def sample_numpy():
    return np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [128, 128, 128]]], dtype=np.uint8)


@pytest.fixture
def sample_keypoints():
    return [cv.KeyPoint(10, 20, 1), cv.KeyPoint(30, 40, 1)]


class TestTensorUtils:
    def test_is_tensor(self):
        assert is_tensor(torch.tensor([1]))
        assert not is_tensor(np.array([1]))
        assert not is_tensor(None)

    def test_empty(self):
        result = empty((2, 3))
        assert result.shape == (2, 3)
        assert result.dtype == torch.float32
        assert result.device.type == "cpu"
        assert empty((0, 3)).numel() == 0

    def test_empty_invalid_shape(self):
        with pytest.raises((TypeError, RuntimeError)):
            empty((-1, 3))

    def test_get_device(self):
        assert get_device("cpu") == torch.device("cpu")

    def test_get_device_invalid(self):
        with pytest.raises((RuntimeError, ValueError)):
            get_device("invalid_device")

    def test_numpy_to_tensor(self, sample_numpy):
        result = numpy_to_tensor(sample_numpy)
        assert result.shape == sample_numpy.shape
        assert result.dtype == torch.uint8
        np.testing.assert_array_equal(result.numpy(), sample_numpy)

    def test_numpy_to_tensor_dtype(self):
        result = numpy_to_tensor(np.array([1, 2]), dtype=torch.float32)
        assert result.dtype == torch.float32

    def test_numpy_to_tensor_non_contiguous(self):
        array = np.arange(12).reshape(3, 4)[:, ::2]
        result = numpy_to_tensor(array)
        assert result.is_contiguous()
        np.testing.assert_array_equal(result.numpy(), array)

    def test_tensor_to_numpy(self):
        tensor = torch.tensor([1.0, 2.0], requires_grad=True)
        result = tensor_to_numpy(tensor)
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_tensor_to_numpy_dtype(self):
        result = tensor_to_numpy(torch.tensor([1.5, 2.5]), dtype=np.uint8)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, [1, 2])

    def test_tensor_to_numpy_non_tensor(self):
        array = np.array([1, 2])
        assert tensor_to_numpy(array) is array
        assert tensor_to_numpy(None) is None


class TestImageConversions:
    def test_rgb_to_bgr(self):
        image = np.array([[[255, 0, 0]]], dtype=np.uint8)
        result = to_numpy_bgr(image)
        np.testing.assert_array_equal(result, [[[0, 0, 255]]])

    def test_grayscale_to_bgr(self):
        result = to_numpy_bgr(np.array([[128]], dtype=np.uint8))
        assert result.shape == (1, 1, 3)
        np.testing.assert_array_equal(result[0, 0], [128, 128, 128])

    def test_tensor_to_bgr(self):
        result = to_numpy_bgr(torch.zeros((3, 2, 2)), input_type="tensor")
        assert result.shape == (2, 2, 3)
        assert result.dtype == np.uint8

    def test_image_to_tensor(self):
        image = np.array([[[255, 0, 0]]], dtype=np.uint8)
        result = image_to_tensor(image)
        assert result.shape == (3, 1, 1)
        assert result.dtype == torch.float32
        torch.testing.assert_close(result[:, 0, 0], torch.tensor([0., 0., 1.]))

    def test_grayscale_to_tensor(self):
        result = image_to_tensor(np.array([[255, 0]], dtype=np.uint8))
        assert result.shape == (3, 1, 2)
        torch.testing.assert_close(
            result[:, 0, :],
            torch.tensor([[1., 0.], [1., 0.], [1., 0.]]),
        )

    def test_image_to_tensor_normalization(self, sample_numpy):
        result = image_to_tensor(sample_numpy)
        assert result.min() >= 0
        assert result.max() <= 1

    def test_image_to_tensor_single_channel(self):
        image = np.array([[[255], [0]], [[128], [64]]], dtype=np.uint8)
        assert image_to_tensor(image).shape == (1, 2, 2)

    def test_image_to_cv(self):
        tensor = torch.tensor([[[1.]], [[0.]], [[0.]]])
        result = image_to_cv(tensor)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result[0, 0], [0, 0, 255])

    def test_image_to_cv_grayscale(self):
        result = image_to_cv(torch.tensor([[0., 1.], [0.5, 0.25]]))
        assert result.shape == (2, 2, 1)
        np.testing.assert_array_equal(
            result[:, :, 0], [[0, 255], [127, 63]]
        )

    def test_image_roundtrip(self, sample_numpy):
        result = image_to_cv(image_to_tensor(sample_numpy))
        np.testing.assert_allclose(result, sample_numpy, atol=1)


class TestFeatureConversions:
    def test_features_to_tensor(self, sample_keypoints):
        data = {
            "keypoints": sample_keypoints,
            "descriptors": np.array([[1., 2.], [3., 4.]], dtype=np.float32),
            "scores": np.array([0.9, 0.8], dtype=np.float32),
            "width": 640,
            "height": 480,
        }
        result = features_to_tensor(data)

        assert result["keypoints"].shape == (2, 2)
        assert result["descriptors"].shape == (2, 2)
        assert result["scores"].shape == (2,)
        assert result["width"] == 640
        assert result["height"] == 480
        torch.testing.assert_close(
            result["keypoints"], torch.tensor([[10., 20.], [30., 40.]])
        )

    def test_features_to_tensor_missing_data(self, sample_keypoints):
        result = features_to_tensor({"keypoints": sample_keypoints})
        assert result["descriptors"].shape == (2, 0)
        assert result["scores"].shape == (2,)

    def test_features_to_tensor_empty(self):
        result = features_to_tensor({"keypoints": []})
        assert result["keypoints"].shape == (0, 2)
        assert result["descriptors"].shape == (0, 0)
        assert result["scores"].shape == (0,)

    def test_features_to_cv(self):
        data = {
            "keypoints": torch.tensor([[10., 20.], [30., 40.]]),
            "descriptors": torch.tensor([[1., 2.], [3., 4.]]),
            "scores": torch.tensor([0.9, 0.8]),
            "width": 640,
            "height": 480,
        }
        result = features_to_cv(data)

        assert all(isinstance(kp, cv.KeyPoint) for kp in result["keypoints"])
        np.testing.assert_allclose(
            cv.KeyPoint_convert(result["keypoints"]),
            [[10., 20.], [30., 40.]],
        )
        np.testing.assert_allclose(result["descriptors"], [[1., 2.], [3., 4.]])
        np.testing.assert_allclose(result["scores"], [0.9, 0.8])
        assert result["width"] == 640
        assert result["height"] == 480

    def test_features_to_cv_batched_keypoints(self):
        data = {"keypoints": torch.tensor([[[1., 2.], [3., 4.]]])}
        assert len(features_to_cv(data)["keypoints"]) == 2

    def test_features_roundtrip(self, sample_keypoints):
        data = {
            "keypoints": sample_keypoints,
            "descriptors": np.array([[1., 2.], [3., 4.]], dtype=np.float32),
            "scores": np.array([0.9, 0.8], dtype=np.float32),
        }
        result = features_to_cv(features_to_tensor(data))

        np.testing.assert_allclose(
            cv.KeyPoint_convert(result["keypoints"]),
            cv.KeyPoint_convert(sample_keypoints),
        )
        np.testing.assert_allclose(result["descriptors"], data["descriptors"])
        np.testing.assert_allclose(result["scores"], data["scores"])


class TestMatchConversions:
    def test_matches_to_tensor(self):
        matches = [cv.DMatch(0, 1, 0.5), cv.DMatch(2, 3, 0.8)]
        result = matches_to_tensor({"matches": matches})["matches"]

        assert result.dtype == torch.long
        torch.testing.assert_close(result, torch.tensor([[0, 1], [2, 3]]))

    @pytest.mark.parametrize("data", [{}, {"matches": None}, {"matches": []}])
    def test_matches_to_tensor_empty(self, data):
        result = matches_to_tensor(data)["matches"]
        assert result.shape == (0, 2)
        assert result.dtype == torch.long

    def test_matches_to_cv(self):
        data = {
            "matches": torch.tensor([[0, 1], [2, 3]]),
            "scores": torch.tensor([0.9, 0.8]),
        }
        result = matches_to_cv(data)

        assert [(m.queryIdx, m.trainIdx) for m in result["matches"]] == [
            (0, 1), (2, 3)
        ]
        np.testing.assert_allclose(result["scores"], [0.9, 0.8])

    def test_matches_to_cv_empty(self):
        result = matches_to_cv({
            "matches": torch.empty((0, 2), dtype=torch.long),
            "scores": torch.empty((0,)),
        })
        assert len(result["matches"]) == 0
        assert result["scores"].shape == (0,)

    def test_matches_roundtrip(self):
        matches = [cv.DMatch(0, 1, 0.5), cv.DMatch(2, 3, 0.8)]
        result = matches_to_cv(matches_to_tensor({"matches": matches}))
        assert [(m.queryIdx, m.trainIdx) for m in result["matches"]] == [
            (0, 1), (2, 3)
        ]

    def test_keypoints_in_matches_to_cv(self):
        data = {
            "matches": torch.tensor([[0, 1]]),
            "keypoints0": torch.tensor([[10., 20.]]),
            "keypoints1": torch.tensor([[30., 40.]]),
        }
        result = matches_to_cv(data)

        np.testing.assert_allclose(cv.KeyPoint_convert(result["keypoints0"]), [[10., 20.]])
        np.testing.assert_allclose(cv.KeyPoint_convert(result["keypoints1"]), [[30., 40.]])
