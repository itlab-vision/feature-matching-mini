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
            upsample_res=(864, 1152)
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
            upsample_res=(160, 160)
        )
        assert result == {"model": mock_model}

    def test_call(self, wrapper, mock_model):
        img0 = MagicMock()
        img1 = MagicMock()
        warp = MagicMock()
        certainty = MagicMock()
        matches = torch.rand(1, 10, 4)
        conf = torch.rand(1, 10)
        kp0 = torch.rand(10, 2)
        kp1 = torch.rand(10, 2)

        mock_model.match.return_value = (warp, certainty)
        mock_model.sample.return_value = (matches, conf)
        mock_model.to_pixel_coordinates.return_value = (kp0, kp1)

        inputs = {
            "image0": img0,
            "image1": img1,
            "height0": 100,
            "width0": 200,
            "height1": 150,
            "width1": 300
        }

        result = wrapper.call(inputs)

        mock_model.match.assert_called_once_with(img0, img1, device=torch.device("cpu"))
        mock_model.sample.assert_called_once_with(warp, certainty, num=10)
        mock_model.to_pixel_coordinates.assert_called_once_with(
            matches, 100, 200, 150, 300
        )

        assert torch.equal(result["keypoints0"], kp0)
        assert torch.equal(result["keypoints1"], kp1)
        assert torch.equal(result["scores"], conf)

    def test_call_error(self, wrapper, mock_model, mock_logger):
        mock_model.match.side_effect = RuntimeError("match failed")

        inputs = {
            "image0": MagicMock(),
            "image1": MagicMock(),
            "height0": 100,
            "width0": 200,
            "height1": 100,
            "width1": 200
        }

        with pytest.raises(RuntimeError, match="match failed"):
            wrapper.call(inputs)

        mock_logger.error.assert_called_once_with("RoMa match error: match failed")