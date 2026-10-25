import pytest
import torch
import numpy as np
from unittest.mock import patch

from src.backend.io_adapter_base import IOAdapter
from src.backend.torch.io_adapter.loftr import LoFTRTorchIOAdapter


@pytest.fixture
def adapter():
    obj = LoFTRTorchIOAdapter.__new__(LoFTRTorchIOAdapter)
    obj._device = torch.device("cpu")
    return obj


class TestLoFTRTorchIOAdapter:
    def test_register(self):
        assert IOAdapter._METHODS["loftr_torch"] == LoFTRTorchIOAdapter

    def test_preprocess_image_color(self, adapter):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[:, :, 0] = 255

        result = adapter._preprocess_image(img)

        assert isinstance(result, torch.Tensor)
        assert result.shape == torch.Size([1, 1, 96, 200])
        assert result.dtype == torch.float32
        assert result.device == adapter._device

    def test_preprocess_image_grayscale(self, adapter):
        img = np.full((80, 120), 128, dtype=np.uint8)

        result = adapter._preprocess_image(img)

        assert result.shape == (1, 1, 80, 120)
        assert result.dtype == torch.float32
        assert torch.allclose(result.mean(), torch.tensor(128 / 255), atol=1e-5)

    def test_preprocess_image_tensor(self, adapter):
        img = torch.ones((3, 80, 120), dtype=torch.float32)

        result = adapter._preprocess_image(img)

        assert result.shape == (1, 1, 80, 120)
        assert result.dtype == torch.float32
        assert torch.allclose(result, torch.ones_like(result))

    def test_preprocess_image_batched_tensor(self, adapter):
        img = torch.ones((1, 3, 80, 120), dtype=torch.float32)

        result = adapter._preprocess_image(img)

        assert result.shape == (1, 1, 80, 120)
        assert torch.allclose(result, torch.ones_like(result))

    def test_preprocess_image_resizes_large_image(self, adapter):
        img = np.full((800, 1200), 255, dtype=np.uint8)

        result = adapter._preprocess_image(img)

        assert result.shape == (1, 1, 424, 640)
        assert result.dtype == torch.float32
        assert torch.allclose(result, torch.ones_like(result))

    def test_preprocess_image_normalizes_values(self, adapter):
        img = np.full((80, 120), 128, dtype=np.uint8)

        result = adapter._preprocess_image(img)

        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)
        assert torch.allclose(result.mean(), torch.tensor(128 / 255), atol=1e-5)

    @patch("src.backend.torch.io_adapter.loftr.image_to_tensor", side_effect=lambda img: img)
    def test_preprocess(self, mock_image_to_tensor, adapter):
        img0 = np.zeros((100, 200, 3), dtype=np.uint8)
        img1 = np.zeros((150, 300, 3), dtype=np.uint8)
        inputs = {
            "image0": img0,
            "image1": img1
        }

        result = adapter.preprocess(inputs)

        assert set(result.keys()) == {"image0", "image1"}
        assert result["image0"].shape == torch.Size([1, 1, 96, 200])
        assert result["image1"].shape == torch.Size([1, 1, 144, 296])
        assert result["image0"].device == adapter._device
        assert result["image1"].device == adapter._device
        assert mock_image_to_tensor.call_count == 2
        assert "image0" not in inputs
        assert "image1" not in inputs

    @patch("src.backend.torch.io_adapter.loftr.matches_to_cv")
    def test_postprocess(self, mock_matches_to_cv, adapter):
        keypoints0 = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
        keypoints1 = torch.tensor([[11.0, 21.0], [31.0, 41.0]])
        scores = torch.tensor([0.9, 0.8])
        mock_matches_to_cv.return_value = {"matches": "converted"}

        outputs = {
            "mkpts0_f": keypoints0,
            "mkpts1_f": keypoints1,
            "mconf": scores
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

    @patch("src.backend.torch.io_adapter.loftr.matches_to_cv")
    def test_postprocess_empty_matches(self, mock_matches_to_cv, adapter):
        outputs = {
            "mkpts0_f": torch.empty((0, 2)),
            "mkpts1_f": torch.empty((0, 2)),
            "mconf": torch.empty((0,))
        }
        mock_matches_to_cv.return_value = {}

        result = adapter.postprocess(outputs)

        data = mock_matches_to_cv.call_args.args[0]
        assert data["matches"].shape == (0, 2)
        assert data["scores"].shape == (0,)
        assert result == {}
