import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch
from logging import Logger

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.torch.model_wrapper.loftr import LoFTRTorchModelWrapper


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
def wrapper(mock_logger, mock_model, tmp_path):
    obj = LoFTRTorchModelWrapper.__new__(LoFTRTorchModelWrapper)
    obj._logger = mock_logger
    obj._device = torch.device("cpu")
    obj._config = {}
    obj._model_path = str(tmp_path / "loftr.ckpt")
    obj._model_name = "loftr"
    obj._model = mock_model
    return obj


class TestLoFTRTorchModelWrapper:
    def test_register(self):
        assert ModelWrapper._METHODS["loftr_torch"] == LoFTRTorchModelWrapper

    @patch("src.backend.torch.model_wrapper.loftr.torch.load")
    @patch("src.backend.torch.model_wrapper.loftr.LoFTRModel")
    def test_load(self, mock_loftr_model, mock_torch_load, wrapper, mock_logger, mock_model):
        checkpoint = Path(wrapper._model_path)
        checkpoint.touch()

        mock_loftr_model.return_value = mock_model
        mock_torch_load.return_value = {"state_dict": {"weight": torch.tensor([1.0])}}

        result = wrapper.load()

        mock_loftr_model.assert_called_once()
        mock_torch_load.assert_called_once_with(str(checkpoint), map_location="cpu")
        mock_model.load_state_dict.assert_called_once_with(
            {"weight": torch.tensor([1.0])}
        )
        mock_model.to.assert_called_once_with(torch.device("cpu"))
        mock_model.eval.assert_called_once_with()

        assert result == {"model": mock_model}
        assert wrapper._weights_type == "outdoor"
        mock_logger.info.assert_any_call("Initializing LoFTR onto cpu")
        mock_logger.info.assert_any_call("LoFTR successfully loaded.")

    @patch("src.backend.torch.model_wrapper.loftr.torch.load")
    @patch("src.backend.torch.model_wrapper.loftr.LoFTRModel")
    def test_load_custom_weights(self, mock_loftr_model, mock_torch_load, wrapper, mock_model):
        checkpoint = Path(wrapper._model_path)
        checkpoint.touch()
        wrapper._config = {"weights": "indoor"}

        mock_loftr_model.return_value = mock_model
        mock_torch_load.return_value = {"state_dict": {}}

        result = wrapper.load()

        assert wrapper._weights_type == "indoor"
        assert result == {"model": mock_model}

    @patch("src.backend.torch.model_wrapper.loftr.torch.hub.download_url_to_file")
    @patch("src.backend.torch.model_wrapper.loftr.torch.load")
    @patch("src.backend.torch.model_wrapper.loftr.LoFTRModel")
    def test_load_downloads_missing_checkpoint(self, mock_loftr_model, mock_torch_load, mock_download, wrapper,
                                               mock_model):
        wrapper._model_path = str(Path(wrapper._model_path).parent / "missing.ckpt")
        mock_loftr_model.return_value = mock_model
        mock_torch_load.return_value = {"state_dict": {}}

        result = wrapper.load()

        mock_download.assert_called_once()
        assert mock_download.call_args.args[1] == wrapper._model_path
        mock_torch_load.assert_called_once_with(
            wrapper._model_path,
            map_location="cpu"
        )
        assert result == {"model": mock_model}

    def test_call(self, wrapper, mock_model):
        inputs = {
            "image0": torch.rand(1, 1, 64, 64),
            "image1": torch.rand(1, 1, 64, 64)
        }

        result = wrapper.call(inputs)

        mock_model.assert_called_once_with(inputs)
        assert result is inputs

    def test_call_error(self, wrapper, mock_model, mock_logger):
        mock_model.side_effect = RuntimeError("inference failed")
        inputs = {
            "image0": torch.rand(1, 1, 64, 64),
            "image1": torch.rand(1, 1, 64, 64)
        }

        with pytest.raises(RuntimeError, match="inference failed"):
            wrapper.call(inputs)

        mock_logger.error.assert_called_once_with("LoFTR inference error: inference failed")