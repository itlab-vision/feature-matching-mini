import pytest
from unittest.mock import MagicMock, patch
from logging import Logger

from src.backend.onnx.model_loader.aliked_loader import AlikedOnnxModelLoader
from src.backend.model_wrapper_base import ModelWrapper


@pytest.fixture
def mock_logger():
    return MagicMock(spec=Logger)


def test_registered_in_factory(mock_logger):
    loader = AlikedOnnxModelLoader(model_name="aliked_onnx", config={}, logger=mock_logger)
    assert loader is not None
    assert "aliked_onnx" in ModelWrapper._METHODS


def test_load_default_path_success(mock_logger):
    config = {}
    loader = AlikedOnnxModelLoader(model_name="aliked_onnx", config=config, logger=mock_logger)

    with patch('src.backend.onnx.model_loader.aliked_loader.OnnxSessionConfiguration') as mock_session_config_cls, \
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
            "Initializing aliked_onnx from models/aliked-n32-top2k-640.onnx (ONNX Runtime)"
        )
        mock_inference_session.assert_called_once_with(
            "models/aliked-n32-top2k-640.onnx",
            sess_options="dummy_sess_options",
            providers=["CPUExecutionProvider"]
        )


def test_load_custom_path_success(mock_logger):
    custom_path = "models/custom_aliked.onnx"
    config = {'aliked_model_path': custom_path}
    loader = AlikedOnnxModelLoader(model_name="aliked_onnx", config=config, logger=mock_logger)

    with patch('src.backend.onnx.model_loader.aliked_loader.OnnxSessionConfiguration') as mock_session_config_cls, \
            patch('onnxruntime.InferenceSession') as mock_inference_session:
        mock_session_config_instance = MagicMock()
        mock_session_config_instance.set_session_options.return_value = "dummy_sess_options"
        mock_session_config_instance.get_providers_with_options.return_value = ["CPUExecutionProvider"]
        mock_session_config_cls.return_value = mock_session_config_instance

        loader.load()

        mock_logger.info.assert_any_call(
            f"Initializing aliked_onnx from {custom_path} (ONNX Runtime)"
        )
        mock_inference_session.assert_called_once_with(
            custom_path,
            sess_options="dummy_sess_options",
            providers=["CPUExecutionProvider"]
        )

        assert 'aliked_model_path' not in config


def test_load_exception_handling(mock_logger):
    config = {}
    loader = AlikedOnnxModelLoader(model_name="aliked_onnx", config=config, logger=mock_logger)

    with patch('src.backend.onnx.model_loader.aliked_loader.OnnxSessionConfiguration'), \
            patch('onnxruntime.InferenceSession', side_effect=Exception("File not found")):
        with pytest.raises(Exception, match="File not found"):
            loader.load()

        mock_logger.error.assert_called_once()
        error_msg = mock_logger.error.call_args[0][0]
        assert "Failed to load ONNX model aliked_onnx" in error_msg
        assert "File not found" in error_msg
