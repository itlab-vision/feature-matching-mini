from unittest.mock import MagicMock

import pytest

from src.backend.inference_api_base import InferenceAPI
from src.backend.executorch.inference_api import ExecuTorchInferenceAPI


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def mock_model_wrapper():
    return MagicMock()


@pytest.fixture
def api(mock_logger, mock_model_wrapper):
    return ExecuTorchInferenceAPI(
        logger=mock_logger,
        model_name="d2net_executorch",
        model_wrapper=mock_model_wrapper,
        config={},
    )


class TestExecuTorchInferenceAPIRegistration:
    def test_registered_in_factory(self):
        assert InferenceAPI._METHODS["executorch"] is ExecuTorchInferenceAPI


class TestExecuTorchInferenceAPIInit:
    def test_accepts_none_config(self, mock_logger, mock_model_wrapper):
        api = ExecuTorchInferenceAPI(
            logger=mock_logger,
            model_name="xfeat_executorch",
            model_wrapper=mock_model_wrapper,
        )

        assert api._config == {}

    def test_preserves_custom_config(self, mock_logger, mock_model_wrapper):
        config = {"num_keypoints": 512}

        api = ExecuTorchInferenceAPI(
            logger=mock_logger,
            model_name="xfeat_executorch",
            model_wrapper=mock_model_wrapper,
            config=config,
        )

        assert api._config is config


class TestExecuTorchInferenceAPIRun:
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
        mock_model_wrapper.call.side_effect = RuntimeError("inference failed")

        with pytest.raises(RuntimeError, match="inference failed"):
            api.run({"image": "test_image"})

    def test_run_passes_empty_inputs(self, api, mock_model_wrapper):
        mock_model_wrapper.call.return_value = {}

        result = api.run({})

        mock_model_wrapper.call.assert_called_once_with({})
        assert result == {}
