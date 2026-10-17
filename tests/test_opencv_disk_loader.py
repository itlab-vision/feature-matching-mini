import pytest
from unittest.mock import MagicMock, patch
from src.backend.opencv.model_loader.disk_loader import DiskOpenCVModelLoader


@pytest.fixture
def mock_logger():
    return MagicMock()


def test_disk_loader_custom_path(mock_logger):
    config = {'disk_model_path': 'models/custom_disk.onnx'}
    loader = DiskOpenCVModelLoader(
        model_name="disk_opencv",
        config=config,
        logger=mock_logger
    )

    with patch('cv2.dnn.readNet') as mock_read_net:
        fake_net = MagicMock()
        mock_read_net.return_value = fake_net

        model = loader.load()

        mock_read_net.assert_called_once_with('models/custom_disk.onnx')

        mock_logger.info.assert_called_once_with("Initializing Disk from models/custom_disk.onnx")

        assert model == fake_net
        assert 'disk_model_path' not in loader._config


def test_disk_loader_default_path(mock_logger):
    config = {}
    loader = DiskOpenCVModelLoader(
        model_name="disk_opencv",
        config=config,
        logger=mock_logger
    )

    with patch('cv2.dnn.readNet') as mock_read_net:
        fake_net = MagicMock()
        mock_read_net.return_value = fake_net

        model = loader.load()

        default_path = "models/disk_1024.onnx"
        mock_read_net.assert_called_once_with(default_path)

        mock_logger.info.assert_called_once_with(f"Initializing Disk from {default_path}")

        assert model == fake_net
