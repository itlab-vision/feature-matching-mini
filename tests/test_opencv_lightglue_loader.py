import pytest
from unittest.mock import MagicMock, patch
from src.backend.opencv.model_loader.lightglue_loader import LightGlueOpenCVModelLoader


@pytest.fixture
def mock_logger():
    return MagicMock()


def test_lightglue_loader_custom_path(mock_logger):
    config = {'lightglue_model_path': 'models/custom_lightglue.onnx'}
    loader = LightGlueOpenCVModelLoader(
        model_name="lightglue_opencv",
        config=config,
        logger=mock_logger
    )

    with patch('cv2.dnn.readNet') as mock_read_net:
        fake_net = MagicMock()
        mock_read_net.return_value = fake_net

        model = loader.load()

        mock_read_net.assert_called_once_with('models/custom_lightglue.onnx')

        mock_logger.info.assert_called_once_with("Initializing LightGlue from models/custom_lightglue.onnx")

        assert model == fake_net
        assert 'lightglue_model_path' not in loader._config


def test_lightglue_loader_default_path(mock_logger):
    config = {}
    loader = LightGlueOpenCVModelLoader(
        model_name="lightglue_opencv",
        config=config,
        logger=mock_logger
    )

    with patch('cv2.dnn.readNet') as mock_read_net:
        fake_net = MagicMock()
        mock_read_net.return_value = fake_net

        model = loader.load()

        default_path = "models/disk_lightglue_2outputs.onnx"
        mock_read_net.assert_called_once_with(default_path)

        mock_logger.info.assert_called_once_with(f"Initializing LightGlue from {default_path}")

        assert model == fake_net
