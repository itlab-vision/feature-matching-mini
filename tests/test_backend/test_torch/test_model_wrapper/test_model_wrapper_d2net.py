from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.torch.model_wrapper.d2net import D2NetTorchModelWrapper


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def wrapper(mock_logger, tmp_path):
    obj = D2NetTorchModelWrapper(model_name="d2net_torch", logger=mock_logger, config={"device": "cpu"})
    obj._model_path = str(tmp_path / "d2net.pth")
    obj._device = torch.device("cpu")
    return obj


class TestD2NetTorchModelWrapperRegistration:
    def test_registered_in_factory(self):
        assert ModelWrapper._METHODS["d2net_torch"] is D2NetTorchModelWrapper


class TestD2NetTorchModelWrapperLoad:
    @patch("src.backend.torch.model_wrapper.d2net.D2NetModel")
    @patch("src.backend.torch.model_wrapper.d2net.torch.hub.download_url_to_file")
    def test_downloads_weights_when_missing(self, mock_download, mock_model, wrapper, tmp_path):
        checkpoint = tmp_path / "weights" / "d2net.pth"
        wrapper._model_path = str(checkpoint)

        result = wrapper.load()

        mock_download.assert_called_once()
        assert mock_download.call_args.args[1] == str(checkpoint)
        assert checkpoint.parent.exists()
        mock_model.assert_called_once_with(
            model_file=str(checkpoint),
            use_relu=True,
            use_cuda=False,
        )
        assert result == {"model": wrapper._model}
        assert wrapper._model.eval.called

    @patch("src.backend.torch.model_wrapper.d2net.D2NetModel")
    @patch("src.backend.torch.model_wrapper.d2net.torch.hub.download_url_to_file")
    def test_does_not_download_existing_weights(self, mock_download, mock_model, wrapper, tmp_path):
        checkpoint = tmp_path / "d2net.pth"
        checkpoint.touch()
        wrapper._model_path = str(checkpoint)

        wrapper.load()

        mock_download.assert_not_called()
        mock_model.assert_called_once()

    @patch("src.backend.torch.model_wrapper.d2net.D2NetModel")
    @patch("src.backend.torch.model_wrapper.d2net.torch.hub.download_url_to_file")
    def test_custom_use_relu(self, mock_download, mock_model, wrapper, tmp_path):
        checkpoint = tmp_path / "d2net.pth"
        checkpoint.touch()
        wrapper._model_path = str(checkpoint)
        wrapper._config["use_relu"] = False

        wrapper.load()

        assert mock_model.call_args.kwargs["use_relu"] is False

    @patch("src.backend.torch.model_wrapper.d2net.D2NetModel")
    @patch("src.backend.torch.model_wrapper.d2net.torch.hub.download_url_to_file")
    def test_cuda_device_enables_cuda(self, mock_download, mock_model, wrapper, tmp_path):
        checkpoint = tmp_path / "d2net.pth"
        checkpoint.touch()
        wrapper._model_path = str(checkpoint)
        wrapper._config["device"] = "cuda:0"

        wrapper.load()

        assert mock_model.call_args.kwargs["use_cuda"] is True

    @patch("src.backend.torch.model_wrapper.d2net.D2NetModel")
    @patch("src.backend.torch.model_wrapper.d2net.torch.hub.download_url_to_file")
    def test_model_is_moved_to_device_and_eval(self, mock_download, mock_model, wrapper, tmp_path):
        checkpoint = tmp_path / "d2net.pth"
        checkpoint.touch()
        wrapper._model_path = str(checkpoint)

        model = mock_model.return_value
        model.to.return_value = model

        wrapper.load()

        model.to.assert_called_once_with(wrapper._device)
        model.eval.assert_called_once()


class TestD2NetTorchModelWrapperCall:
    @patch("src.backend.torch.model_wrapper.d2net.process_multiscale")
    def test_call_preprocesses_image_and_returns_numpy_features(self, process, wrapper):
        process.return_value = (np.array([[1, 2, 3]]), np.array([0.8]), np.ones((1, 4)))

        result = wrapper.call({"image": np.zeros((8, 10, 3), dtype=np.uint8)})

        tensor, model = process.call_args.args
        assert tensor.shape == (1, 3, 8, 10)
        assert model is wrapper._model
        assert process.call_args.kwargs == {"scales": [1]}
        np.testing.assert_array_equal(result["keypoints"], [[2, 1]])
        np.testing.assert_array_equal(result["descriptors"], np.ones((1, 4)))
        np.testing.assert_array_equal(result["scores"], [0.8])

    @patch("src.backend.torch.model_wrapper.d2net.process_multiscale", side_effect=RuntimeError("boom"))
    def test_call_logs_and_reraises_inference_error(self, process, wrapper, mock_logger):
        with pytest.raises(RuntimeError, match="boom"):
            wrapper.call({"image": np.zeros((8, 10, 3), dtype=np.uint8)})
        mock_logger.error.assert_called_once_with("D2-Net inference error: boom")

    def test_call_missing_image_reraises_error(self, wrapper):
        with pytest.raises(KeyError, match="image"):
            wrapper.call({})
