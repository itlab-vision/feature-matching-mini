import pytest
from unittest.mock import MagicMock
from src.backend.opencv.inference_api.inference_api import OpenCVInferenceAPI


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def mock_model_wrapper():
    model = MagicMock()
    model.forward.return_value = ["tensor1", "tensor2", "tensor3"]
    return model


@pytest.mark.parametrize("model_name,expected_outputs", [
    ("opencv", None),
    ("aliked_opencv", ["keypoints", "descriptors", "scores"]),
    ("disk_opencv", ["keypoints", "descriptors", "scores"]),
    ("lightglue_opencv", ["matches0", "mscores0"])
])
def test_inference_api_initialization_and_output_names(mock_logger, mock_model_wrapper, model_name, expected_outputs):
    api = OpenCVInferenceAPI(
        logger=mock_logger,
        model_name=model_name,
        model=mock_model_wrapper,
        config={}
    )
    assert OpenCVInferenceAPI is not None
    assert api._model_name == model_name
    assert api.output_names == expected_outputs


def test_run_success(mock_logger, mock_model_wrapper):
    api = OpenCVInferenceAPI(
        logger=mock_logger,
        model_name="aliked_opencv",
        model=mock_model_wrapper
    )

    outputs = api.run()

    mock_model_wrapper.forward.assert_called_once_with(["keypoints", "descriptors", "scores"])
    assert outputs == ["tensor1", "tensor2", "tensor3"]
    mock_logger.error.assert_not_called()


def test_run_exception_handling(mock_logger, mock_model_wrapper):
    runtime_error = RuntimeError("OpenCV C++ exception")
    mock_model_wrapper.forward.side_effect = runtime_error

    api = OpenCVInferenceAPI(
        logger=mock_logger,
        model_name="disk_opencv",
        model=mock_model_wrapper
    )

    outputs = api.run()

    mock_logger.error.assert_called_once()
    error_msg = mock_logger.error.call_args[0][0]
    assert "inference error" in error_msg
    assert outputs is None
