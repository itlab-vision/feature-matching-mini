import pytest
from unittest.mock import MagicMock, patch
from logging import Logger

from src.backend.onnx.model_loader.lightglue_loader import LightGlueOnnxModelLoader
from src.backend.model_wrapper_base import ModelWrapper


@pytest.fixture
def mock_logger():
    return MagicMock(spec=Logger)


def test_registered_in_factory(mock_logger):
    loader = LightGlueOnnxModelLoader(model_name="lightglue_onnx", config={}, logger=mock_logger)
    assert loader is not None
    assert "lightglue_onnx" in ModelWrapper._METHODS


def test_load_default_path_success(mock_logger):
    config = {}
    loader = LightGlueOnnxModelLoader(model_name="lightglue_onnx", config=config, logger=mock_logger)

    with patch('src.backend.onnx.model_loader.lightglue_loader.OnnxSessionConfiguration') as mock_session_config_cls, \
            patch('onnxruntime.InferenceSession') as mock_inference_session:
        mock_session_config_instance = MagicMock()
        mock_session_config_instance.set_session_options.return_value = "dummy_sess_options"
        mock_session_config_instance.get_providers_with_options.return_value = ["CPUExecutionProvider"]
        mock_session_config_cls.return_value = mock_session_config_instance

        mock_session_instance = MagicMock()
        mock_inference_session.return_value = mock_session_instance

        model = loader.load()['model']

        assert model == mock_session_instance
        mock_logger.info.assert_any_call(
            "Initializing lightglue_onnx from models/disk_lightglue_2outputs.onnx (ONNX Runtime)"
        )
        mock_inference_session.assert_called_once_with(
            "models/disk_lightglue_2outputs.onnx",
            sess_options="dummy_sess_options",
            providers=["CPUExecutionProvider"]
        )


def test_load_custom_path_success(mock_logger):
    custom_path = "models/custom_lightglue.onnx"
    config = {'lightglue_model_path': custom_path}
    loader = LightGlueOnnxModelLoader(model_name="lightglue_onnx", config=config, logger=mock_logger)

    with patch('src.backend.onnx.model_loader.lightglue_loader.OnnxSessionConfiguration') as mock_session_config_cls, \
            patch('onnxruntime.InferenceSession') as mock_inference_session:
        mock_session_config_instance = MagicMock()
        mock_session_config_instance.set_session_options.return_value = "dummy_sess_options"
        mock_session_config_instance.get_providers_with_options.return_value = ["CPUExecutionProvider"]
        mock_session_config_cls.return_value = mock_session_config_instance

        loader.load()

        mock_logger.info.assert_any_call(
            f"Initializing lightglue_onnx from {custom_path} (ONNX Runtime)"
        )
        mock_inference_session.assert_called_once_with(
            custom_path,
            sess_options="dummy_sess_options",
            providers=["CPUExecutionProvider"]
        )

        assert 'lightglue_model_path' not in config


def test_load_exception_handling(mock_logger):
    config = {}
    loader = LightGlueOnnxModelLoader(model_name="lightglue_onnx", config=config, logger=mock_logger)

    with patch('src.backend.onnx.model_loader.lightglue_loader.OnnxSessionConfiguration'), \
            patch('onnxruntime.InferenceSession', side_effect=Exception("LightGlue model not found")):
        with pytest.raises(Exception, match="LightGlue model not found"):
            loader.load()

        mock_logger.error.assert_called_once()
        error_msg = mock_logger.error.call_args[0][0]
        assert "Failed to load ONNX model lightglue_onnx" in error_msg
        assert "LightGlue model not found" in error_msg
