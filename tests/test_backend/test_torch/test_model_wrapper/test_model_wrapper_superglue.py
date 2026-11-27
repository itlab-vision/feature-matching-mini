import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch
from logging import Logger

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.torch.model_wrapper.superglue import SuperGlueTorchModelWrapper


@pytest.fixture
def mock_logger():
    return MagicMock(spec=Logger)


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model
    return model


@pytest.fixture
def wrapper(mock_logger, mock_model):
    obj = SuperGlueTorchModelWrapper.__new__(SuperGlueTorchModelWrapper)
    obj._logger = mock_logger
    obj._device = torch.device("cpu")
    obj._config = {}
    obj._model = mock_model
    return obj


class TestSuperGlueTorchModelWrapper:
    def test_register(self):
        assert ModelWrapper._METHODS["superglue_torch"] == SuperGlueTorchModelWrapper

    @patch("src.backend.torch.model_wrapper.superglue.SuperGlue")
    def test_load_default_config(self, mock_superglue, wrapper, mock_logger, mock_model):
        mock_superglue.return_value.to.return_value.eval.return_value = mock_model
        wrapper._model = None

        result = wrapper.load()

        mock_superglue.assert_called_once_with({
            "weights": "outdoor",
            "sinkhorn_iterations": 20,
            "match_threshold": 0.005
        })
        mock_superglue.return_value.to.assert_called_once_with(torch.device("cpu"))
        mock_superglue.return_value.to.return_value.eval.assert_called_once_with()
        assert result == {"model": mock_model}
        mock_logger.info.assert_called_once_with("Loading SuperGlue (outdoor) onto cpu")

    @patch("src.backend.torch.model_wrapper.superglue.SuperGlue")
    def test_load_custom_config(self, mock_superglue, wrapper, mock_model):
        wrapper._config = {
            "weights": "indoor",
            "sinkhorn_iterations": 10,
            "threshold": 0.1
        }
        mock_superglue.return_value.to.return_value.eval.return_value = mock_model

        result = wrapper.load()
        mock_superglue.assert_called_once_with({
            "weights": "indoor",
            "sinkhorn_iterations": 10,
            "match_threshold": 0.1
        })
        assert result == {"model": mock_model}

    def test_call(self, wrapper, mock_model):
        features = {
            "keypoints": np.ones((3, 2), dtype=np.float32),
            "descriptors": np.ones((3, 4), dtype=np.float32),
            "scores": np.ones(3, dtype=np.float32),
            "height": 8, "width": 10,
        }
        mock_model.return_value = {
            "matches0": torch.tensor([[2, -1, 3]]),
            "matching_scores0": torch.tensor([[0.75, 0.0, 0.5]]),
        }

        result = wrapper.call({"features0": features, "features1": features})

        data = mock_model.call_args.args[0]
        assert data["descriptors0"].shape == (1, 4, 3)
        assert data["image1"].shape == (1, 1, 8, 10)
        np.testing.assert_array_equal(result["matches"], [[0, 2]])
        np.testing.assert_array_equal(result["scores"], [0.75])

    def test_call_error(self, wrapper, mock_logger):
        with pytest.raises(KeyError, match="features0"):
            wrapper.call({})
        mock_logger.error.assert_called_once()
