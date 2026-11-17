import pytest
import torch
from unittest.mock import MagicMock, patch
from logging import Logger

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.torch.model_wrapper.roma import RoMaTorchModelWrapper


@pytest.fixture
def mock_logger():
    return MagicMock(spec=Logger)


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.eval.return_value = model
    model.to.return_value = model
    return model


@pytest.fixture
def wrapper(mock_logger, mock_model):
    obj = RoMaTorchModelWrapper.__new__(RoMaTorchModelWrapper)
    obj._logger = mock_logger
    obj._device = torch.device("cpu")
    obj._config = {}
    obj._nfeatures = 10
    obj._model = mock_model
    return obj


class TestRoMaTorchModelWrapper:
    def test_register(self):
        assert ModelWrapper._METHODS["roma_torch"] == RoMaTorchModelWrapper

    @patch("src.backend.torch.model_wrapper.roma.roma_outdoor")
    def test_load_default_config(self, mock_roma_outdoor, wrapper, mock_logger, mock_model):
        mock_roma_outdoor.return_value = mock_model

        result = wrapper.load()

        mock_roma_outdoor.assert_called_once_with(
            device=torch.device("cpu"),
            coarse_res=560,
            upsample_res=(864, 1152),
            use_custom_corr=False,
        )
        mock_model.eval.assert_called_once_with()
        mock_model.to.assert_called_once_with(torch.device("cpu"))
        assert result == {"model": mock_model}
        mock_logger.info.assert_called_once_with("Loading RoMa weights onto cpu")

    @patch("src.backend.torch.model_wrapper.roma.roma_outdoor")
    def test_load_custom_config(self, mock_roma_outdoor, wrapper, mock_model):
        wrapper._config = {
            "coarse_res": 128,
            "upsample_res": (160, 160)
        }
        mock_roma_outdoor.return_value = mock_model

        result = wrapper.load()

        mock_roma_outdoor.assert_called_once_with(
            device=torch.device("cpu"),
            coarse_res=128,
            upsample_res=(160, 160),
            use_custom_corr=False,
        )
        assert result == {"model": mock_model}
