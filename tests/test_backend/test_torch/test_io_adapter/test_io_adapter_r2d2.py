import numpy as np
import pytest
import torch

from unittest.mock import patch

from src.backend.io_adapter_base import IOAdapter
from src.backend.torch.io_adapter.r2d2 import R2D2TorchIOAdapter


@pytest.fixture
def adapter():
    obj = R2D2TorchIOAdapter.__new__(R2D2TorchIOAdapter)
    obj._device = torch.device("cpu")
    return obj


@pytest.fixture
def random_img():
    return np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)


@pytest.fixture
def model_outputs():
    return {
        "keypoints": torch.rand(100, 3),
        "descriptors": torch.rand(100, 128),
        "scores": torch.rand(100),
    }


class TestR2D2TorchIOAdapterRegistration:
    def test_registered_in_factory(self):
        assert "r2d2_torch" in IOAdapter._METHODS
        assert IOAdapter._METHODS["r2d2_torch"] is R2D2TorchIOAdapter


class TestR2D2TorchIOAdapterPreprocess:
    @patch("src.backend.torch.io_adapter.r2d2.image_to_tensor")
    def test_preprocess_returns_dict(self, mock_image_to_tensor, adapter, random_img):
        mock_image_to_tensor.return_value = random_img

        result = adapter.preprocess({"image": random_img})
        assert isinstance(result, dict)
        assert "image" in result

    @patch("src.backend.torch.io_adapter.r2d2.image_to_tensor")
    def test_preprocess_returns_tensor(self, mock_image_to_tensor, adapter, random_img):
        mock_image_to_tensor.return_value = random_img

        result = adapter.preprocess({"image": random_img})
        assert isinstance(result["image"], torch.Tensor)

    @patch("src.backend.torch.io_adapter.r2d2.image_to_tensor")
    def test_preprocess_tensor_shape(self, mock_image_to_tensor, adapter, random_img):
        mock_image_to_tensor.return_value = random_img
        result = adapter.preprocess({"image": random_img})
        assert result["image"].shape == (1, 3, 240, 320)

    @patch("src.backend.torch.io_adapter.r2d2.image_to_tensor")
    def test_preprocess_normalizes_image(self, mock_image_to_tensor, adapter):
        image = np.full((10, 20, 3), 128, dtype=np.uint8)
        mock_image_to_tensor.return_value = image

        result = adapter.preprocess({"image": image})
        tensor = result["image"]

        assert tensor.dtype == torch.float32
        assert tensor.shape == (1, 3, 10, 20)
        assert torch.isfinite(tensor).all()

    @patch("src.backend.torch.io_adapter.r2d2.image_to_tensor")
    def test_preprocess_calls_image_to_tensor(self, mock_image_to_tensor, adapter, random_img):
        mock_image_to_tensor.return_value = random_img
        adapter.preprocess({"image": random_img})
        mock_image_to_tensor.assert_called_once_with(random_img)

    @patch("src.backend.torch.io_adapter.r2d2.image_to_tensor")
    def test_preprocess_converts_bgr_to_rgb(self, mock_image_to_tensor, adapter):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        image[:] = [10, 20, 30]
        mock_image_to_tensor.return_value = image

        result = adapter.preprocess({"image": image})
        tensor = result["image"]
        expected = torch.tensor([
            (30 / 255.0 - 0.485) / 0.229,
            (20 / 255.0 - 0.456) / 0.224,
            (10 / 255.0 - 0.406) / 0.225,
        ])

        assert torch.allclose(tensor[0, :, 0, 0], expected, atol=1e-6)

    @patch("src.backend.torch.io_adapter.r2d2.image_to_tensor")
    def test_preprocess_uses_configured_device(self, mock_image_to_tensor, random_img):
        obj = R2D2TorchIOAdapter.__new__(R2D2TorchIOAdapter)
        obj._device = torch.device("cpu")
        mock_image_to_tensor.return_value = random_img
        result = obj.preprocess({"image": random_img})
        assert result["image"].device.type == "cpu"

    @patch("src.backend.torch.io_adapter.r2d2.image_to_tensor")
    def test_preprocess_preserves_other_inputs(self, mock_image_to_tensor, adapter, random_img):
        metadata = {"id": 123}
        mock_image_to_tensor.return_value = random_img

        result = adapter.preprocess({
            "image": random_img,
            "metadata": metadata,
        })
        assert result["metadata"] is metadata


class TestR2D2TorchIOAdapterPostprocess:
    @patch("src.backend.torch.io_adapter.r2d2.features_to_cv")
    def test_postprocess_calls_features_to_cv(self, mock_features_to_cv, adapter, model_outputs):
        mock_features_to_cv.return_value = model_outputs
        result = adapter.postprocess(model_outputs)

        mock_features_to_cv.assert_called_once_with(model_outputs)
        assert result is model_outputs

    @patch("src.backend.torch.io_adapter.r2d2.features_to_cv")
    def test_postprocess_selects_first_two_keypoint_columns(self, mock_features_to_cv, adapter, model_outputs):
        keypoints = torch.tensor([
            [10.0, 20.0, 0.9],
            [30.0, 40.0, 0.8],
        ])
        outputs = {
            "keypoints": keypoints,
            "descriptors": torch.rand(2, 128),
            "scores": torch.rand(2),
        }
        mock_features_to_cv.side_effect = lambda value: value
        result = adapter.postprocess(outputs)
        assert result["keypoints"].shape == (2, 2)
        assert torch.equal(result["keypoints"], keypoints[:, :2])

    @patch("src.backend.torch.io_adapter.r2d2.features_to_cv")
    def test_postprocess_returns_converted_features(self, mock_features_to_cv, adapter, model_outputs):
        expected = {"keypoints": np.zeros((100, 2))}
        mock_features_to_cv.return_value = expected
        result = adapter.postprocess(model_outputs)
        assert result is expected
