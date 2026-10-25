import pytest
import torch
import numpy as np
from unittest.mock import patch

from src.backend.io_adapter_base import IOAdapter
from src.backend.torch.io_adapter.superglue import SuperGlueTorchIOAdapter


@pytest.fixture
def adapter():
    obj = SuperGlueTorchIOAdapter.__new__(SuperGlueTorchIOAdapter)
    obj._device = torch.device("cpu")
    return obj


@pytest.fixture
def features0():
    return {
        "keypoints": np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        "descriptors": np.array([[3.0, 4.0], [5.0, 12.0]], dtype=np.float32),
        "scores": np.array([0.9, 0.8], dtype=np.float32),
        "height": 100,
        "width": 200
    }


@pytest.fixture
def features1():
    return {
        "keypoints": np.array([[15.0, 25.0], [35.0, 45.0], [55.0, 65.0]], dtype=np.float32),
        "descriptors": np.array([[8.0, 6.0], [7.0, 24.0], [5.0, 12.0]], dtype=np.float32),
        "scores": np.array([0.7, 0.6, 0.5], dtype=np.float32),
        "height": 120,
        "width": 240
    }


class TestSuperGlueTorchIOAdapter:
    def test_register(self):
        assert IOAdapter._METHODS["superglue_torch"] == SuperGlueTorchIOAdapter

    @patch("src.backend.torch.io_adapter.superglue.features_to_tensor", side_effect=lambda x: x)
    def test_preprocess_features(self, mock_features_to_tensor, adapter, features0):
        result = adapter._preprocess_features(features0)

        assert result["keypoints"].shape == (1, 2, 2)
        assert result["descriptors"].shape == (1, 2, 2)
        assert result["scores"].shape == (1, 2)
        assert result["image"].shape == (1, 1, 100, 200)

        assert result["keypoints"].device == adapter._device
        assert result["descriptors"].device == adapter._device
        assert result["scores"].device == adapter._device
        assert result["image"].device == adapter._device

        expected_descriptors = torch.tensor(
            [[3.0, 4.0], [5.0, 12.0]]
        ).float()
        expected_descriptors = torch.nn.functional.normalize(
            expected_descriptors, p=2, dim=1
        ).T.unsqueeze(0)

        assert torch.allclose(result["descriptors"], expected_descriptors)
        mock_features_to_tensor.assert_called_once_with(features0)

    @patch("src.backend.torch.io_adapter.superglue.features_to_tensor", side_effect=lambda x: x)
    def test_preprocess_features_with_tensors(self, mock_features_to_tensor, adapter):
        features = {
            "keypoints": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "descriptors": torch.tensor([[3.0, 4.0], [5.0, 12.0]]),
            "scores": torch.tensor([0.9, 0.8]),
            "height": 50,
            "width": 60
        }

        result = adapter._preprocess_features(features)

        assert result["keypoints"].shape == (1, 2, 2)
        assert result["descriptors"].shape == (1, 2, 2)
        assert result["scores"].shape == (1, 2)
        assert result["image"].shape == (1, 1, 50, 60)

    @patch("src.backend.torch.io_adapter.superglue.features_to_tensor", side_effect=lambda x: x)
    def test_preprocess(self, mock_features_to_tensor, adapter, features0, features1):
        inputs = {
            "features0": features0,
            "features1": features1
        }

        result = adapter.preprocess(inputs)

        assert set(result.keys()) == {
            "keypoints0",
            "keypoints1",
            "descriptors0",
            "descriptors1",
            "scores0",
            "scores1",
            "image0",
            "image1"
        }

        assert result["keypoints0"].shape == (1, 2, 2)
        assert result["keypoints1"].shape == (1, 3, 2)
        assert result["descriptors0"].shape == (1, 2, 2)
        assert result["descriptors1"].shape == (1, 2, 3)
        assert result["scores0"].shape == (1, 2)
        assert result["scores1"].shape == (1, 3)
        assert result["image0"].shape == (1, 1, 100, 200)
        assert result["image1"].shape == (1, 1, 120, 240)

        assert mock_features_to_tensor.call_count == 2

    @patch("src.backend.torch.io_adapter.superglue.matches_to_cv")
    def test_postprocess(self, mock_matches_to_cv, adapter):
        outputs = {
            "matches": torch.tensor([[2, -1, 0, 5, 1]]),
            "scores": torch.tensor([[0.9, 0.0, 0.8, 0.0, 0.7]]),
            "keypoints1": torch.rand(1, 3, 2)
        }
        mock_matches_to_cv.return_value = {"matches": "converted"}

        result = adapter.postprocess(outputs)

        mock_matches_to_cv.assert_called_once()
        matches_and_scores = mock_matches_to_cv.call_args.args[0]

        assert torch.equal(
            matches_and_scores["matches"],
            torch.tensor([[0, 2], [2, 0], [4, 1]])
        )
        assert torch.equal(
            matches_and_scores["scores"],
            torch.tensor([0.9, 0.8, 0.7])
        )
        assert result == {"matches": "converted"}

    @patch("src.backend.torch.io_adapter.superglue.matches_to_cv")
    def test_postprocess_no_valid_matches(self, mock_matches_to_cv, adapter):
        outputs = {
            "matches": torch.tensor([[-1, 5, -1]]),
            "scores": torch.tensor([[0.0, 0.0, 0.0]]),
            "keypoints1": torch.rand(1, 3, 2)
        }
        mock_matches_to_cv.return_value = {}

        result = adapter.postprocess(outputs)

        matches_and_scores = mock_matches_to_cv.call_args.args[0]
        assert matches_and_scores["matches"].shape == (0, 2)
        assert matches_and_scores["scores"].shape == (0,)
        assert result == {}
