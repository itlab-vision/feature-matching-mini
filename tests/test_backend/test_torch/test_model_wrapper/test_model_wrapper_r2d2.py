import pytest
import torch

from unittest.mock import MagicMock, patch
from logging import Logger

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.torch.model_wrapper.r2d2 import R2D2TorchModelWrapper


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
def mock_nms():
    return MagicMock()


@pytest.fixture
def wrapper(mock_logger, mock_model, mock_nms):
    obj = R2D2TorchModelWrapper.__new__(R2D2TorchModelWrapper)
    obj._logger = mock_logger
    obj._device = torch.device("cpu")
    obj._config = {}
    obj._model_path = None
    obj._model = mock_model
    obj._nms = mock_nms
    return obj


@pytest.fixture
def model_outputs():
    return (torch.rand(100, 3), torch.rand(100, 128), torch.rand(100))


class TestR2D2TorchModelWrapperRegistration:
    def test_registered_in_factory(self):
        assert "r2d2_torch" in ModelWrapper._METHODS
        assert ModelWrapper._METHODS["r2d2_torch"] is R2D2TorchModelWrapper


class TestR2D2TorchModelWrapperLoad:
    @patch("src.backend.torch.model_wrapper.r2d2.NonMaxSuppression")
    @patch("src.backend.torch.model_wrapper.r2d2.load_network")
    def test_load_creates_model(self, mock_load_network, mock_nms_class,
                                wrapper, mock_model, mock_nms):
        mock_load_network.return_value = mock_model
        mock_nms_class.return_value = mock_nms

        result = wrapper.load()

        assert result == {"model": mock_model, "nms": mock_nms}
        assert wrapper._model is mock_model
        assert wrapper._nms is mock_nms

    @patch("src.backend.torch.model_wrapper.r2d2.NonMaxSuppression")
    @patch("src.backend.torch.model_wrapper.r2d2.load_network")
    def test_load_uses_custom_checkpoint(self, mock_load_network, mock_nms_class, wrapper,
                                         mock_model, mock_nms, tmp_path):
        checkpoint = tmp_path / "custom_model.pt"
        wrapper._config = {"checkpoint": str(checkpoint)}
        mock_load_network.return_value = mock_model
        mock_nms_class.return_value = mock_nms

        wrapper.load()

        mock_load_network.assert_called_once_with(checkpoint)

    @patch("src.backend.torch.model_wrapper.r2d2.NonMaxSuppression")
    @patch("src.backend.torch.model_wrapper.r2d2.load_network")
    def test_load_prefers_model_path(self, mock_load_network, mock_nms_class, wrapper,
                                     mock_model, mock_nms, tmp_path):
        model_path = tmp_path / "model_path.pt"
        wrapper._model_path = model_path
        wrapper._config = {"checkpoint": "config_checkpoint.pt"}
        mock_load_network.return_value = mock_model
        mock_nms_class.return_value = mock_nms

        wrapper.load()
        mock_load_network.assert_called_once_with(model_path)

    @patch("src.backend.torch.model_wrapper.r2d2.NonMaxSuppression")
    @patch("src.backend.torch.model_wrapper.r2d2.load_network")
    def test_load_moves_model_to_device(self, mock_load_network, mock_nms_class, wrapper,
                                        mock_model, mock_nms):
        mock_load_network.return_value = mock_model
        mock_nms_class.return_value = mock_nms

        wrapper.load()
        mock_model.to.assert_called_once_with(wrapper._device)

    @patch("src.backend.torch.model_wrapper.r2d2.NonMaxSuppression")
    @patch("src.backend.torch.model_wrapper.r2d2.load_network")
    def test_load_sets_eval_mode(self, mock_load_network, mock_nms_class, wrapper,
                                 mock_model, mock_nms):
        mock_load_network.return_value = mock_model
        mock_nms_class.return_value = mock_nms

        wrapper.load()
        mock_model.eval.assert_called_once_with()

    @patch("src.backend.torch.model_wrapper.r2d2.NonMaxSuppression")
    @patch("src.backend.torch.model_wrapper.r2d2.load_network")
    def test_load_creates_nms(self, mock_load_network, mock_nms_class, wrapper,
                              mock_model, mock_nms):
        mock_load_network.return_value = mock_model
        mock_nms_class.return_value = mock_nms

        result = wrapper.load()
        mock_nms_class.assert_called_once_with()
        assert result["nms"] is mock_nms

    @patch("src.backend.torch.model_wrapper.r2d2.NonMaxSuppression")
    @patch("src.backend.torch.model_wrapper.r2d2.load_network")
    def test_load_logs_message(self, mock_load_network, mock_nms_class, wrapper,
                               mock_model, mock_nms):
        mock_load_network.return_value = mock_model
        mock_nms_class.return_value = mock_nms

        wrapper.load()
        wrapper._logger.info.assert_called_once_with(
            "Loading R2D2 weights from "
            "thirdparty/r2d2/models/r2d2_WASF_N16.pt"
        )
