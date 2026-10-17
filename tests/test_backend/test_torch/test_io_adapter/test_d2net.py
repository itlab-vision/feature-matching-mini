import cv2 as cv
import numpy as np
import pytest
from unittest.mock import patch

from src.backend.io_adapter_base import IOAdapter
from src.backend.torch.io_adapter.d2net import D2NetTorchIOAdapter


@pytest.fixture
def adapter():
    return D2NetTorchIOAdapter("d2net_torch")


@pytest.fixture
def image():
    return np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)


class TestD2NetTorchIOAdapterRegistration:
    def test_registered_in_factory(self):
        assert IOAdapter._METHODS["d2net_torch"] is D2NetTorchIOAdapter


class TestD2NetTorchIOAdapterPreprocess:
    @patch("src.backend.torch.io_adapter.d2net.preprocess_image")
    def test_preprocess_calls_caffe_preprocessing(self, mock_preprocess, adapter, image):
        mock_preprocess.return_value = "processed_image"
        result = adapter.preprocess({"image": image})

        mock_preprocess.assert_called_once()
        args, kwargs = mock_preprocess.call_args
        assert kwargs["preprocessing"] == "caffe"
        assert "image" in result
        assert result["image"] == "processed_image"

    @patch("src.backend.torch.io_adapter.d2net.preprocess_image")
    def test_preprocess_removes_image_from_original_inputs(self, mock_preprocess, adapter, image):
        mock_preprocess.return_value = "processed_image"
        inputs = {"image": image, "extra": "value"}

        result = adapter.preprocess(inputs)
        assert "image" not in inputs
        assert result["extra"] == "value"

    @patch("src.backend.torch.io_adapter.d2net.preprocess_image")
    def test_preprocess_preserves_extra_inputs(self, mock_preprocess, adapter, image):
        mock_preprocess.return_value = "processed_image"
        result = adapter.preprocess({"image": image, "threshold": 0.01})
        assert result["threshold"] == 0.01

    def test_preprocess_without_image_raises(self, adapter):
        with pytest.raises(KeyError):
            adapter.preprocess({})


class TestD2NetTorchIOAdapterPostprocess:
    def test_postprocess_converts_xy_keypoints_to_cv_keypoints(self, adapter):
        outputs = {
            "keypoints": np.array([[10, 20], [30, 40]], dtype=np.float32),
            "descriptors": np.random.rand(2, 512).astype(np.float32),
            "scores": np.array([0.9, 0.8], dtype=np.float32),
        }

        result = adapter.postprocess(outputs)

        assert isinstance(result["keypoints"], np.ndarray)
        assert result["keypoints"].shape == (2,)
        assert all(isinstance(kp, cv.KeyPoint) for kp in result["keypoints"])
        assert result["keypoints"][0].pt == (20.0, 10.0)
        assert result["keypoints"][1].pt == (40.0, 30.0)

    def test_postprocess_preserves_descriptors_and_scores(self, adapter):
        descriptors = np.random.rand(2, 512).astype(np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        outputs = {
            "keypoints": np.array([[10, 20], [30, 40]], dtype=np.float32),
            "descriptors": descriptors,
            "scores": scores,
        }

        result = adapter.postprocess(outputs)
        np.testing.assert_array_equal(result["descriptors"], descriptors)
        np.testing.assert_array_equal(result["scores"], scores)

    def test_postprocess_flattens_batched_keypoints(self, adapter):
        outputs = {
            "keypoints": np.array([[[10, 20], [30, 40]]], dtype=np.float32),
            "descriptors": np.random.rand(2, 512).astype(np.float32),
            "scores": np.array([0.9, 0.8], dtype=np.float32),
        }

        result = adapter.postprocess(outputs)
        assert result["keypoints"].shape == (2,)
        assert result["keypoints"][0].pt == (30.0, 40.0)
        assert result["keypoints"][1].pt == (10.0, 20.0)

    def test_postprocess_empty_keypoints(self, adapter):
        outputs = {
            "keypoints": np.empty((0, 2), dtype=np.float32),
            "descriptors": np.empty((0, 512), dtype=np.float32),
            "scores": np.empty((0,), dtype=np.float32),
        }

        result = adapter.postprocess(outputs)

        assert isinstance(result["keypoints"], np.ndarray)
        assert len(result["keypoints"]) == 0
        assert result["descriptors"].shape == (0, 512)
        assert len(result["scores"]) == 0

    def test_postprocess_missing_keypoints(self, adapter):
        outputs = {
            "descriptors": np.empty((0, 512), dtype=np.float32),
            "scores": np.empty((0,), dtype=np.float32),
        }

        with pytest.raises((TypeError, IndexError)):
            adapter.postprocess(outputs)
