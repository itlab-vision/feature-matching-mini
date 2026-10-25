from unittest.mock import MagicMock, patch

import pytest

from src.backend.inference_api_base import InferenceAPI
from src.backend.torch.inference_api import TorchInferenceAPI


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def mock_model_wrapper():
    return MagicMock()


@pytest.fixture
def api(mock_logger, mock_model_wrapper):
    return TorchInferenceAPI(
        logger=mock_logger,
        model_name="d2net_torch",
        model_wrapper=mock_model_wrapper,
        config={},
    )


class TestTorchInferenceAPIRegistration:
    def test_registered_in_factory(self):
        assert InferenceAPI._METHODS["torch"] is TorchInferenceAPI


class TestTorchInferenceAPIInit:
    def test_default_config(self, mock_logger, mock_model_wrapper):
        api = TorchInferenceAPI(
            logger=mock_logger,
            model_name="d2net_torch",
            model_wrapper=mock_model_wrapper,
        )

        assert api._nfeatures == 4096
        assert api._threshold == 0.005

    def test_custom_config(self, mock_logger, mock_model_wrapper):
        api = TorchInferenceAPI(
            logger=mock_logger,
            model_name="d2net_torch",
            model_wrapper=mock_model_wrapper,
            config={
                "device": "cpu",
                "nfeatures": 1024,
                "threshold": 0.01,
            },
        )

        assert api._device == "cpu"
        assert api._nfeatures == 1024
        assert api._threshold == 0.01

    def test_empty_config(self, mock_logger, mock_model_wrapper):
        api = TorchInferenceAPI(
            logger=mock_logger,
            model_name="d2net_torch",
            model_wrapper=mock_model_wrapper,
            config={},
        )

        assert api._nfeatures == 4096
        assert api._threshold == 0.005

    @patch("src.backend.torch.inference_api.inference_api.get_device")
    def test_uses_default_device_when_not_configured(self, mock_get_device, mock_logger, mock_model_wrapper):
        mock_get_device.return_value = "cpu"

        api = TorchInferenceAPI(
            logger=mock_logger,
            model_name="d2net_torch",
            model_wrapper=mock_model_wrapper,
            config={},
        )

        mock_get_device.assert_called_once_with()
        assert api._device == "cpu"

    @patch("src.backend.torch.inference_api.inference_api.get_device")
    def test_does_not_call_get_device_when_device_configured(self, mock_get_device, mock_logger, mock_model_wrapper):
        api = TorchInferenceAPI(
            logger=mock_logger,
            model_name="d2net_torch",
            model_wrapper=mock_model_wrapper,
            config={"device": "cuda"},
        )

        mock_get_device.assert_not_called()
        assert api._device == "cuda"

    def test_zero_nfeatures_is_preserved(self, mock_logger, mock_model_wrapper):
        api = TorchInferenceAPI(
            logger=mock_logger,
            model_name="d2net_torch",
            model_wrapper=mock_model_wrapper,
            config={"nfeatures": 0},
        )

        assert api._nfeatures == 0

    def test_zero_threshold_is_preserved(self, mock_logger, mock_model_wrapper):
        api = TorchInferenceAPI(
            logger=mock_logger,
            model_name="d2net_torch",
            model_wrapper=mock_model_wrapper,
            config={"threshold": 0},
        )

        assert api._threshold == 0


class TestTorchInferenceAPIRun:
    def test_run_calls_model_wrapper(self, api, mock_model_wrapper):
        inputs = {"image": "test_image"}
        expected = {"keypoints": [], "descriptors": []}
        mock_model_wrapper.call.return_value = expected

        result = api.run(inputs)

        mock_model_wrapper.call.assert_called_once_with(inputs)
        assert result is expected

    def test_run_returns_none_when_wrapper_returns_none(self, api, mock_model_wrapper):
        mock_model_wrapper.call.return_value = None

        result = api.run({"image": "test_image"})

        assert result is None

    def test_run_propagates_wrapper_exception(self, api, mock_model_wrapper):
        error = RuntimeError("inference failed")
        mock_model_wrapper.call.side_effect = error

        with pytest.raises(RuntimeError, match="inference failed") as exc:
            api.run({"image": "test_image"})

        assert exc.value is error

    def test_run_passes_empty_inputs(self, api, mock_model_wrapper):
        mock_model_wrapper.call.return_value = {}

        result = api.run({})

        mock_model_wrapper.call.assert_called_once_with({})
        assert result == {}
