import numpy as np
import pytest
import torch

from unittest.mock import MagicMock, patch

from src.backend.io_adapter_base import IOAdapter
from src.backend.torch.io_adapter.superpoint import SuperPointTorchIOAdapter


@pytest.fixture
def adapter():
    obj = SuperPointTorchIOAdapter.__new__(SuperPointTorchIOAdapter)
    obj._device = torch.device("cpu")
    return obj


@pytest.fixture
def random_img():
    return np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)


@pytest.fixture
def mock_processor():
    processor = MagicMock()
    processor.return_value.to.return_value = {"pixel_values": torch.rand(1, 3, 240, 320)}
    return processor


@pytest.fixture
def processed_features():
    return {
        "keypoints": torch.rand(100, 2),
        "descriptors": torch.rand(100, 256),
        "scores": torch.rand(100),
    }


class TestSuperPointTorchIOAdapterRegistration:
    def test_registered_in_factory(self):
        assert "superpoint_torch" in IOAdapter._METHODS
        assert IOAdapter._METHODS["superpoint_torch"] is SuperPointTorchIOAdapter


class TestSuperPointTorchIOAdapterPreprocess:
    @patch("src.backend.torch.io_adapter.superpoint.image_to_tensor")
    @patch("src.backend.torch.io_adapter.superpoint.to_numpy_bgr")
    def test_preprocess_returns_dict(self, mock_to_numpy_bgr, mock_image_to_tensor,
                                     adapter, random_img, mock_processor):
        mock_image_to_tensor.return_value = random_img
        mock_to_numpy_bgr.return_value = random_img

        result = adapter.preprocess({
            "image": random_img,
            "processor": mock_processor,
        })

        assert isinstance(result, dict)
        assert "image" in result
        assert "height" in result
        assert "width" in result
        assert "processor" in result

    @patch("src.backend.torch.io_adapter.superpoint.image_to_tensor")
    @patch("src.backend.torch.io_adapter.superpoint.to_numpy_bgr")
    def test_preprocess_returns_image_dimensions(self, mock_to_numpy_bgr, mock_image_to_tensor, adapter,
                                                 random_img, mock_processor):
        mock_image_to_tensor.return_value = random_img
        mock_to_numpy_bgr.return_value = random_img

        result = adapter.preprocess({
            "image": random_img,
            "processor": mock_processor,
        })

        assert result["height"] == 240
        assert result["width"] == 320

    @patch("src.backend.torch.io_adapter.superpoint.image_to_tensor")
    @patch("src.backend.torch.io_adapter.superpoint.to_numpy_bgr")
    def test_preprocess_calls_image_to_tensor(self, mock_to_numpy_bgr, mock_image_to_tensor, adapter,
                                              random_img, mock_processor):
        mock_image_to_tensor.return_value = random_img
        mock_to_numpy_bgr.return_value = random_img

        adapter.preprocess({
            "image": random_img,
            "processor": mock_processor,
        })

        mock_image_to_tensor.assert_called_once_with(random_img)

    @patch("src.backend.torch.io_adapter.superpoint.image_to_tensor")
    @patch("src.backend.torch.io_adapter.superpoint.to_numpy_bgr")
    def test_preprocess_calls_to_numpy_bgr(self, mock_to_numpy_bgr,
                                           mock_image_to_tensor, adapter,
                                           random_img, mock_processor):
        mock_image_to_tensor.return_value = random_img
        mock_to_numpy_bgr.return_value = random_img

        adapter.preprocess({
            "image": random_img,
            "processor": mock_processor,
        })

        mock_to_numpy_bgr.assert_called_once_with(
            random_img,
            input_type="numpy",
        )

    @patch("src.backend.torch.io_adapter.superpoint.image_to_tensor")
    @patch("src.backend.torch.io_adapter.superpoint.to_numpy_bgr")
    def test_preprocess_calls_processor(self, mock_to_numpy_bgr,
                                        mock_image_to_tensor, adapter,
                                        random_img, mock_processor):
        mock_image_to_tensor.return_value = random_img
        mock_to_numpy_bgr.return_value = random_img

        adapter.preprocess({"image": random_img, "processor": mock_processor})

        mock_processor.assert_called_once_with(random_img, return_tensors="pt")
        mock_processor.return_value.to.assert_called_once_with(adapter._device)

    @patch("src.backend.torch.io_adapter.superpoint.image_to_tensor")
    @patch("src.backend.torch.io_adapter.superpoint.to_numpy_bgr")
    def test_preprocess_preserves_other_inputs(self, mock_to_numpy_bgr,
                                               mock_image_to_tensor, adapter,
                                               random_img, mock_processor):
        metadata = {"id": 123}
        mock_image_to_tensor.return_value = random_img
        mock_to_numpy_bgr.return_value = random_img

        result = adapter.preprocess({
            "image": random_img,
            "processor": mock_processor,
            "metadata": metadata,
        })

        assert result["metadata"] is metadata


class TestSuperPointTorchIOAdapterPostprocess:
    def test_postprocess_calls_processor(self, adapter, processed_features):
        processor = MagicMock()
        inference_outputs = MagicMock()
        processor.post_process_keypoint_detection.return_value = [processed_features]

        outputs = {
            "processor": processor,
            "inference_outputs": inference_outputs,
            "height": 240,
            "width": 320,
        }

        with patch("src.backend.torch.io_adapter.superpoint.features_to_cv", return_value=processed_features):
            adapter.postprocess(outputs)

        processor.post_process_keypoint_detection.assert_called_once_with(inference_outputs, [[240, 320]])

    def test_postprocess_calls_features_to_cv(self, adapter, processed_features):
        processor = MagicMock()
        processor.post_process_keypoint_detection.return_value = [
            processed_features
        ]

        outputs = {
            "processor": processor,
            "inference_outputs": MagicMock(),
            "height": 240,
            "width": 320,
        }

        with (patch("src.backend.torch.io_adapter.superpoint.features_to_cv", return_value=processed_features)
              as mock_features_to_cv):
            adapter.postprocess(outputs)

        mock_features_to_cv.assert_called_once_with(outputs)

    def test_postprocess_returns_converted_features(self, adapter, processed_features):
        processor = MagicMock()
        processor.post_process_keypoint_detection.return_value = [processed_features]

        outputs = {
            "processor": processor,
            "inference_outputs": MagicMock(),
            "height": 240,
            "width": 320,
        }

        with patch("src.backend.torch.io_adapter.superpoint.features_to_cv", return_value=processed_features):
            result = adapter.postprocess(outputs)

        assert result is processed_features

    def test_postprocess_updates_feature_fields(self, adapter, processed_features):
        processor = MagicMock()
        processor.post_process_keypoint_detection.return_value = [processed_features]

        outputs = {
            "processor": processor,
            "inference_outputs": MagicMock(),
            "height": 240,
            "width": 320,
        }

        with patch("src.backend.torch.io_adapter.superpoint.features_to_cv", side_effect=lambda value: value):
            result = adapter.postprocess(outputs)

        assert result["keypoints"] is processed_features["keypoints"]
        assert result["descriptors"] is processed_features["descriptors"]
        assert result["scores"] is processed_features["scores"]
