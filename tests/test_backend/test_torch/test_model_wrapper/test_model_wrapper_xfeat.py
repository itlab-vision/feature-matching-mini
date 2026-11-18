import pytest
import torch

from unittest.mock import MagicMock, patch
from logging import Logger

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.torch.model_wrapper.xfeat import XFeatTorchModelWrapper


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
    obj = XFeatTorchModelWrapper.__new__(XFeatTorchModelWrapper)
    obj._logger = mock_logger
    obj._device = torch.device("cpu")
    obj._nfeatures = 256
    obj._model = mock_model
    return obj


@pytest.fixture
def model_output():
    return {
        "keypoints": torch.rand(1, 100, 2),
        "descriptors": torch.rand(1, 100, 64),
        "scores": torch.rand(1, 100),
    }


class TestXFeatTorchModelWrapperRegistration:
    def test_registered_in_factory(self):
        assert "xfeat_torch" in ModelWrapper._METHODS
        assert (ModelWrapper._METHODS["xfeat_torch"] is XFeatTorchModelWrapper)


class TestXFeatTorchModelWrapperLoad:
    @patch("src.backend.torch.model_wrapper.xfeat.XFeatModel")
    def test_load_creates_model(self, mock_xfeat_model, wrapper, mock_model):
        mock_xfeat_model.return_value = mock_model

        result = wrapper.load()

        mock_xfeat_model.assert_called_once_with()
        assert result == {"model": mock_model}
        assert wrapper._model is mock_model

    @patch("src.backend.torch.model_wrapper.xfeat.XFeatModel")
    def test_load_moves_model_to_device(self, mock_xfeat_model, wrapper, mock_model):
        mock_xfeat_model.return_value = mock_model
        wrapper.load()
        mock_model.to.assert_called_once_with(wrapper._device)

    @patch("src.backend.torch.model_wrapper.xfeat.XFeatModel")
    def test_load_sets_model_device_attribute(self, mock_xfeat_model, wrapper, mock_model):
        mock_xfeat_model.return_value = mock_model
        wrapper.load()
        assert mock_model.dev == wrapper._device

    @patch("src.backend.torch.model_wrapper.xfeat.XFeatModel")
    def test_load_sets_eval_mode(self, mock_xfeat_model, wrapper, mock_model):
        mock_xfeat_model.return_value = mock_model
        wrapper.load()
        mock_model.eval.assert_called_once_with()

    @patch("src.backend.torch.model_wrapper.xfeat.XFeatModel")
    def test_load_logs_message(self, mock_xfeat_model, wrapper, mock_model):
        mock_xfeat_model.return_value = mock_model
        wrapper.load()
        wrapper._logger.info.assert_called_once_with(f"Loading XFeat weights onto {wrapper._device}")
