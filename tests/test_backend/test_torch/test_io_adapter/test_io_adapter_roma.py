import pytest
import torch
import numpy as np
from PIL import Image
from unittest.mock import patch

from src.backend.io_adapter_base import IOAdapter
from src.backend.torch.io_adapter.roma import RoMaTorchIOAdapter


@pytest.fixture
def adapter():
    obj = RoMaTorchIOAdapter.__new__(RoMaTorchIOAdapter)
    obj._device = torch.device("cpu")
    return obj


class TestRoMaTorchIOAdapter:
    def test_register(self):
        assert IOAdapter._METHODS["roma_torch"] == RoMaTorchIOAdapter

    def test_preprocess_image_color(self, adapter):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[:, :, 0] = 255

        result = adapter._preprocess_image(img)

        assert isinstance(result, Image.Image)
        assert result.size == (200, 100)
        assert result.mode == "RGB"
        assert result.getpixel((0, 0)) == (0, 0, 255)

    def test_preprocess_image_grayscale(self, adapter):
        img = np.full((50, 80), 128, dtype=np.uint8)

        result = adapter._preprocess_image(img)

        assert isinstance(result, Image.Image)
        assert result.size == (80, 50)
        assert result.mode == "RGB"
        assert result.getpixel((0, 0)) == (128, 128, 128)

    def test_preprocess_image_tensor(self, adapter):
        img = torch.zeros((3, 40, 60), dtype=torch.uint8)
        img[0, :, :] = 255

        result = adapter._preprocess_image(img)

        assert isinstance(result, Image.Image)
        assert result.size == (60, 40)
        assert result.mode == "RGB"
        assert result.getpixel((0, 0)) == (255, 0, 0)

    def test_preprocess_image_batched_tensor(self, adapter):
        img = torch.zeros((1, 3, 40, 60), dtype=torch.uint8)
        img[:, 1, :, :] = 255

        result = adapter._preprocess_image(img)

        assert isinstance(result, Image.Image)
        assert result.size == (60, 40)
        assert result.getpixel((0, 0)) == (0, 255, 0)

    def test_preprocess_image_float_tensor(self, adapter):
        img = torch.zeros((3, 20, 30), dtype=torch.float32)
        img[0, :, :] = 1.0

        result = adapter._preprocess_image(img)

        assert isinstance(result, Image.Image)
        assert result.size == (30, 20)
        assert result.getpixel((0, 0)) == (255, 0, 0)

    @patch("src.backend.torch.io_adapter.roma.image_to_tensor", side_effect=lambda img: img)
    def test_preprocess(self, mock_image_to_tensor, adapter):
        img0 = np.zeros((100, 200, 3), dtype=np.uint8)
        img1 = np.zeros((150, 300, 3), dtype=np.uint8)
        inputs = {
            "image0": img0,
            "image1": img1,
            "extra": "value"
        }

        result = adapter.preprocess(inputs)

        assert isinstance(result["image0"], Image.Image)
        assert isinstance(result["image1"], Image.Image)
        assert result["image0"].size == (200, 100)
        assert result["image1"].size == (300, 150)
        assert result["width0"] == 200
        assert result["height0"] == 100
        assert result["width1"] == 300
        assert result["height1"] == 150
        assert result["extra"] == "value"
        assert "image0" not in inputs
        assert "image1" not in inputs
        assert mock_image_to_tensor.call_count == 2

    @patch("src.backend.torch.io_adapter.roma.matches_to_cv")
    def test_postprocess(self, mock_matches_to_cv, adapter):
        keypoints0 = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
        keypoints1 = torch.tensor([[11.0, 21.0], [31.0, 41.0]])
        scores = torch.tensor([0.9, 0.8])
        mock_matches_to_cv.return_value = {"matches": "converted"}

        outputs = {
            "keypoints0": keypoints0,
            "keypoints1": keypoints1,
            "scores": scores
        }

        result = adapter.postprocess(outputs)

        mock_matches_to_cv.assert_called_once()
        data = mock_matches_to_cv.call_args.args[0]

        assert torch.equal(data["keypoints0"], keypoints0)
        assert torch.equal(data["keypoints1"], keypoints1)
        assert torch.equal(data["scores"], scores)
        assert torch.equal(
            data["matches"],
            torch.tensor([[0, 0], [1, 1]])
        )
        assert result == {"matches": "converted"}

    @patch("src.backend.torch.io_adapter.roma.matches_to_cv")
    def test_postprocess_empty_matches(self, mock_matches_to_cv, adapter):
        outputs = {
            "keypoints0": torch.empty((0, 2)),
            "keypoints1": torch.empty((0, 2)),
            "scores": torch.empty((0,))
        }
        mock_matches_to_cv.return_value = {"matches": []}

        result = adapter.postprocess(outputs)

        data = mock_matches_to_cv.call_args.args[0]
        assert data["matches"].shape == (0, 2)
        assert data["scores"].shape == (0,)
        assert result == {"matches": []}
