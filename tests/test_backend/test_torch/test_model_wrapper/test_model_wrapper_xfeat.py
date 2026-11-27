import numpy as np
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


class TestXFeatTorchModelWrapperCall:
    def test_call_converts_image_and_returns_numpy_features(self, wrapper, model_output):
        def infer(image, top_k):
            assert not torch.is_grad_enabled()
            assert image.shape == (1, 3, 8, 10)
            assert top_k == 256
            return [model_output]

        wrapper._model.detectAndCompute.side_effect = infer
        result = wrapper.call({"image": np.zeros((8, 10, 3), dtype=np.uint8)})

        assert set(result) == set(model_output)
        for key, value in result.items():
            np.testing.assert_array_equal(value, model_output[key].numpy())

    def test_call_logs_and_reraises_model_error(self, wrapper):
        wrapper._model.detectAndCompute.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            wrapper.call({"image": np.zeros((8, 10, 3), dtype=np.uint8)})
        wrapper._logger.warning.assert_called_once_with("XFeat inference error: boom")
