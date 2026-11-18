import numpy as np
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

    def test_call(self, wrapper, mock_model):
        images = {"image0": np.zeros((8, 10, 3), dtype=np.uint8), "image1": np.zeros((7, 9), dtype=np.uint8)}
        mock_model.match.return_value = ("warp", "certainty")
        scores = torch.ones(2)
        matches = torch.zeros(2, 4)
        mock_model.sample.return_value = (matches, scores)
        points = torch.tensor([[2, 3], [4, 5]])
        mock_model.to_pixel_coordinates.return_value = (points, points + 1)

        result = wrapper.call(images)

        left, right = mock_model.match.call_args.args
        assert (left.size, right.size) == ((10, 8), (9, 7))
        assert left.mode == right.mode == "RGB"
        mock_model.sample.assert_called_once_with("warp", "certainty", num=10)
        mock_model.to_pixel_coordinates.assert_called_once_with(matches, 8, 10, 7, 9)
        np.testing.assert_array_equal(result["matches"], [[0, 0], [1, 1]])
        np.testing.assert_array_equal(result["keypoints0"], points.numpy())
        np.testing.assert_array_equal(result["keypoints1"], (points + 1).numpy())
        np.testing.assert_array_equal(result["scores"], scores.numpy())

    def test_call_error(self, wrapper, mock_model, mock_logger):
        mock_model.match.side_effect = RuntimeError("boom")
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="boom"):
            wrapper.call({"image0": image, "image1": image})
        mock_logger.error.assert_called_once_with("RoMa match error: boom")
