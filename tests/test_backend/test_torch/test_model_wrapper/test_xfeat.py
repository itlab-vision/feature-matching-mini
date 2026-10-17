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
    def test_call_returns_dict(self, wrapper, model_output):
        wrapper._model.detectAndCompute.return_value = [model_output]

        result = wrapper.call({"image": torch.rand(1, 3, 120, 160)})

        assert isinstance(result, dict)
        assert "keypoints" in result
        assert "descriptors" in result
        assert "scores" in result

    def test_call_returns_model_outputs(self, wrapper, model_output):
        wrapper._model.detectAndCompute.return_value = [model_output]

        result = wrapper.call({"image": torch.rand(1, 3, 120, 160)})

        assert result["keypoints"] is model_output["keypoints"]
        assert result["descriptors"] is model_output["descriptors"]
        assert result["scores"] is model_output["scores"]

    def test_call_invokes_detect_and_compute(self, wrapper, model_output):
        image = torch.rand(1, 3, 120, 160)
        wrapper._model.detectAndCompute.return_value = [model_output]

        wrapper.call({"image": image})

        wrapper._model.detectAndCompute.assert_called_once_with(image, top_k=wrapper._nfeatures)

    def test_call_uses_configured_nfeatures(self, wrapper, model_output):
        wrapper._nfeatures = 512
        wrapper._model.detectAndCompute.return_value = [model_output]

        wrapper.call({"image": torch.rand(1, 3, 120, 160)})

        wrapper._model.detectAndCompute.assert_called_once_with(
            wrapper._model.detectAndCompute.call_args.args[0],
            top_k=512)

    def test_call_runs_without_gradients(self, wrapper, model_output):
        wrapper._model.detectAndCompute.return_value = [model_output]
        image = torch.rand(1, 3, 120, 160)

        with patch("src.backend.torch.model_wrapper.xfeat.torch.no_grad") as mock_no_grad:
            mock_no_grad.return_value.__enter__ = MagicMock(return_value=None)
            mock_no_grad.return_value.__exit__ = MagicMock(return_value=False)

            wrapper.call({"image": image})

        mock_no_grad.assert_called_once_with()

    def test_call_logs_warning_on_error(self, wrapper):
        error = RuntimeError("inference failed")
        wrapper._model.detectAndCompute.side_effect = error

        with pytest.raises(RuntimeError, match="inference failed"):
            wrapper.call({"image": torch.rand(1, 3, 120, 160)})

        wrapper._logger.warning.assert_called_once_with("XFeat inference error: inference failed")

    def test_call_reraises_model_error(self, wrapper):
        error = ValueError("invalid image")
        wrapper._model.detectAndCompute.side_effect = error

        with pytest.raises(ValueError, match="invalid image"):
            wrapper.call({"image": torch.rand(1, 3, 120, 160)})
