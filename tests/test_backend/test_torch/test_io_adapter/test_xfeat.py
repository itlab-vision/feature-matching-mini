import numpy as np
import pytest
import torch

from unittest.mock import patch

from src.backend.io_adapter_base import IOAdapter
from src.backend.torch.io_adapter.xfeat import XFeatTorchIOAdapter


@pytest.fixture
def adapter():
    obj = XFeatTorchIOAdapter.__new__(XFeatTorchIOAdapter)
    obj._device = torch.device("cpu")
    return obj


@pytest.fixture
def random_img():
    return np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)


@pytest.fixture
def mock_features():
    return {
        "keypoints": torch.rand(1, 100, 2),
        "descriptors": torch.rand(1, 100, 64),
        "scores": torch.rand(1, 100),
    }


class TestXFeatTorchIOAdapterRegistration:

    def test_registered_in_factory(self):
        assert "xfeat_torch" in IOAdapter._METHODS
        assert IOAdapter._METHODS["xfeat_torch"] is XFeatTorchIOAdapter


class TestXFeatTorchIOAdapterPreprocess:

    def test_preprocess_returns_dict(self, adapter, random_img):
        with patch(
            f"{XFeatTorchIOAdapter.__module__}.to_numpy_bgr",
            return_value=random_img,
        ):
            result = adapter.preprocess({"image": random_img})

        assert isinstance(result, dict)
        assert "image" in result

    def test_preprocess_returns_tensor(self, adapter, random_img):
        with patch(
            f"{XFeatTorchIOAdapter.__module__}.to_numpy_bgr",
            return_value=random_img,
        ):
            result = adapter.preprocess({"image": random_img})

        assert isinstance(result["image"], torch.Tensor)

    def test_preprocess_tensor_shape(self, adapter, random_img):
        with patch(
            f"{XFeatTorchIOAdapter.__module__}.to_numpy_bgr",
            return_value=random_img,
        ):
            result = adapter.preprocess({"image": random_img})

        tensor = result["image"]
        assert tensor.shape == (1, 3, 120, 160)

    def test_preprocess_normalizes_image(self, adapter):
        img = np.full((10, 20, 3), 255, dtype=np.uint8)

        with patch(
            f"{XFeatTorchIOAdapter.__module__}.to_numpy_bgr",
            return_value=img,
        ):
            result = adapter.preprocess({"image": img})

        tensor = result["image"]
        assert tensor.dtype == torch.float32
        assert torch.allclose(tensor, torch.ones_like(tensor))

    def test_preprocess_converts_bgr_to_rgb(self, adapter):
        img = np.zeros((2, 2, 3), dtype=np.uint8)
        img[:, :] = [10, 20, 30]

        with patch(
            f"{XFeatTorchIOAdapter.__module__}.to_numpy_bgr",
            return_value=img,
        ):
            result = adapter.preprocess({"image": img})

        tensor = result["image"]

        assert tensor[0, 0, 0, 0].item() == pytest.approx(30 / 255)
        assert tensor[0, 1, 0, 0].item() == pytest.approx(20 / 255)
        assert tensor[0, 2, 0, 0].item() == pytest.approx(10 / 255)

    def test_preprocess_calls_to_numpy_bgr_for_numpy_input(
        self, adapter, random_img
    ):
        with patch(
            f"{XFeatTorchIOAdapter.__module__}.to_numpy_bgr",
            return_value=random_img,
        ) as mock_convert:
            adapter.preprocess({"image": random_img})

        mock_convert.assert_called_once_with(random_img, input_type="numpy")

    def test_preprocess_calls_to_numpy_bgr_for_tensor_input(
        self, adapter, random_img
    ):
        image_tensor = torch.from_numpy(random_img)

        with patch(
            f"{XFeatTorchIOAdapter.__module__}.to_numpy_bgr",
            return_value=random_img,
        ) as mock_convert:
            adapter.preprocess({"image": image_tensor})

        mock_convert.assert_called_once_with(image_tensor, input_type="torch")

    def test_preprocess_preserves_other_inputs(self, adapter, random_img):
        metadata = {"id": 123, "name": "test"}

        with patch(
            f"{XFeatTorchIOAdapter.__module__}.to_numpy_bgr",
            return_value=random_img,
        ):
            result = adapter.preprocess({
                "image": random_img,
                "metadata": metadata,
            })

        assert result["metadata"] is metadata

    def test_preprocess_uses_configured_device(self, random_img):
        obj = XFeatTorchIOAdapter.__new__(XFeatTorchIOAdapter)
        obj._device = torch.device("cpu")

        with patch(
            f"{XFeatTorchIOAdapter.__module__}.to_numpy_bgr",
            return_value=random_img,
        ):
            result = obj.preprocess({"image": random_img})

        assert result["image"].device.type == "cpu"


class TestXFeatTorchIOAdapterPostprocess:
    def test_postprocess_calls_features_to_cv(self, adapter, mock_features):
        expected = {
            "keypoints": np.array([[10.0, 20.0]], dtype=np.float32),
            "descriptors": np.ones((1, 64), dtype=np.float32),
            "scores": np.ones(1, dtype=np.float32),
        }

        with patch(
            f"{XFeatTorchIOAdapter.__module__}.features_to_cv",
            return_value=expected,
        ) as mock_convert:
            result = adapter.postprocess(mock_features)

        mock_convert.assert_called_once_with(mock_features)
        assert result is expected

    def test_postprocess_returns_dict(self, adapter, mock_features):
        expected = {"keypoints": [], "descriptors": [], "scores": []}

        with patch(
            f"{XFeatTorchIOAdapter.__module__}.features_to_cv",
            return_value=expected,
        ):
            result = adapter.postprocess(mock_features)

        assert isinstance(result, dict)

    def test_postprocess_preserves_converted_features(self, adapter, mock_features):
        expected = {
            "keypoints": np.array([[10, 20]], dtype=np.float32),
            "descriptors": np.zeros((1, 64), dtype=np.float32),
            "scores": np.array([0.9], dtype=np.float32),
        }

        with patch(
            f"{XFeatTorchIOAdapter.__module__}.features_to_cv",
            return_value=expected,
        ):
            result = adapter.postprocess(mock_features)

        assert result["keypoints"] is expected["keypoints"]
        assert result["descriptors"] is expected["descriptors"]
        assert result["scores"] is expected["scores"]
