import pytest
from unittest.mock import MagicMock, patch
from src.backend.opencv.model_loader.aliked_loader import AlikedOpenCVModelLoader


@pytest.fixture
def mock_logger():
    return MagicMock()


def test_aliked_loader_custom_path(mock_logger):
    config = {'aliked_model_path': 'models/custom_aliked.onnx'}
    loader = AlikedOpenCVModelLoader(
        model_name="aliked_opencv",
        config=config,
        logger=mock_logger
    )

    with patch('cv2.dnn.readNet') as mock_read_net:
        fake_net = MagicMock()
        mock_read_net.return_value = fake_net

        model = loader.load()

        mock_read_net.assert_called_once_with('models/custom_aliked.onnx')

        mock_logger.info.assert_called_once_with("Initializing Aliked from models/custom_aliked.onnx")

        assert model == fake_net
        assert 'aliked_model_path' not in loader._config


def test_aliked_loader_default_path(mock_logger):
    config = {}
    loader = AlikedOpenCVModelLoader(
        model_name="aliked_opencv",
        config=config,
        logger=mock_logger
    )

    with patch('cv2.dnn.readNet') as mock_read_net:
        fake_net = MagicMock()
        mock_read_net.return_value = fake_net

        model = loader.load()

        default_path = "models/aliked-n32-top2k-640.onnx"
        mock_read_net.assert_called_once_with(default_path)

        mock_logger.info.assert_called_once_with(f"Initializing Aliked from {default_path}")

        assert model == fake_net
