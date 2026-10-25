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
        inputs = {
            "keypoints0": torch.rand(1, 5, 2),
            "keypoints1": torch.rand(1, 6, 2),
            "descriptors0": torch.rand(1, 64, 5),
            "descriptors1": torch.rand(1, 64, 6),
            "scores0": torch.rand(1, 5),
            "scores1": torch.rand(1, 6),
            "image0": torch.rand(1, 1, 100, 100),
            "image1": torch.rand(1, 1, 120, 120)
        }
        matches = torch.tensor([[1, -1, 0, 3, -1]])
        scores = torch.tensor([[0.9, 0.0, 0.8, 0.7, 0.0]])
        mock_model.return_value = {
            "matches0": matches,
            "matching_scores0": scores
        }

        result = wrapper.call(inputs)

        mock_model.assert_called_once_with(inputs)
        assert torch.equal(result["matches"], matches)
        assert torch.equal(result["scores"], scores)
        for key, value in inputs.items():
            assert result[key] is value

    def test_call_error(self, wrapper, mock_model, mock_logger):
        inputs = {"keypoints0": torch.rand(1, 5, 2)}
        mock_model.side_effect = RuntimeError("inference failed")

        with pytest.raises(RuntimeError, match="inference failed"):
            wrapper.call(inputs)

        mock_logger.error.assert_called_once_with("Super Glue inference error: inference failed")
