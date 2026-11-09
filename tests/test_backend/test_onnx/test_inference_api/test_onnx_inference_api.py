import numpy as np
import pytest
from unittest.mock import MagicMock
from src.backend.onnx.inference_api.inference_api import OnnxInferenceAPI


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def mock_model_wrapper():
    model = MagicMock()
    model.run.return_value = ["tensor1", "tensor2", "tensor3"]
    return model


@pytest.mark.parametrize("model_name,expected_outputs", [
    ("onnx", None),
    ("aliked", ["keypoints", "descriptors", "scores"]),
    ("disk", ["keypoints", "descriptors", "scores"]),
    ("lightglue", ["matches0", "mscores0"])
])
def test_inference_api_initialization_and_output_names(mock_logger, mock_model_wrapper, model_name, expected_outputs):
    api = OnnxInferenceAPI(
        logger=mock_logger,
        model_name=model_name,
        model=mock_model_wrapper,
        config={}
    )
    assert OnnxInferenceAPI is not None
    assert api._model_name == model_name
    assert api.output_names == expected_outputs


def test_run_success(mock_logger, mock_model_wrapper):
    api = OnnxInferenceAPI(
        logger=mock_logger,
        model_name="aliked",
        model=mock_model_wrapper
    )

    inputs = {"image": np.zeros((480, 640, 3), dtype=np.uint8)}
    outputs = api.run(inputs)

    assert mock_model_wrapper.run.call_count == 1
    call_args = mock_model_wrapper.run.call_args[0]
    assert call_args[0] == ["keypoints", "descriptors", "scores"]
    assert "image" in call_args[1]
    assert outputs == ["tensor1", "tensor2", "tensor3"]
    mock_logger.error.assert_not_called()


def test_run_exception_handling(mock_logger, mock_model_wrapper):
    runtime_error = RuntimeError("ONNX Runtime execution exception")
    mock_model_wrapper.run.side_effect = runtime_error

    api = OnnxInferenceAPI(
        logger=mock_logger,
        model_name="disk",
        model=mock_model_wrapper
    )

    outputs = api.run(inputs={"image": "dummy"})

    mock_logger.error.assert_called_once()
    error_msg = mock_logger.error.call_args[0][0]
    assert "inference error" in error_msg
    assert outputs is None
