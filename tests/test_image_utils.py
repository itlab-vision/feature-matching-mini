import pytest
import cv2 as cv
import numpy as np
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil

from feature_matching.image_utils import read_image, save_image, show_image, to_numpy_bgr
from feature_matching.converter import Converter


@pytest.fixture
def temp_dir():
    dir_path = tempfile.mkdtemp()
    yield Path(dir_path)
    shutil.rmtree(dir_path)


@pytest.fixture
def test_image_path(temp_dir):
    img_path = temp_dir / "test_image.jpg"
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    cv.imwrite(str(img_path), test_img)
    return img_path


@pytest.fixture
def sample_numpy():
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_tensor():
    return torch.rand(3, 100, 100)


@pytest.fixture
def sample_tensor_grayscale():
    return torch.rand(1, 100, 100)


class TestReadImage:
    def test_read_numpy(self, test_image_path):
        img = read_image(test_image_path, input_type='numpy')
        assert isinstance(img, np.ndarray)
        assert img.shape == (100, 100, 3)

    def test_read_tensor(self, test_image_path):
        img = read_image(test_image_path, input_type='tensor')
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 100, 100)
        assert img.max() <= 1.0
        assert img.min() >= 0.0

    def test_read_none_path(self):
        with pytest.raises(ValueError, match="Empty path"):
            read_image(None)

    def test_read_nonexistent_path(self):
        with pytest.raises(ValueError, match="Incorrect path"):
            read_image(Path("/nonexistent.jpg"))


class TestSaveImage:
    def test_save_numpy(self, sample_numpy, temp_dir):
        save_path = temp_dir / "output.jpg"
        result = save_image(sample_numpy, save_path, input_type='numpy')

        assert result is True
        assert save_path.exists()

    def test_save_tensor(self, sample_tensor, temp_dir):
        save_path = temp_dir / "output.jpg"
        result = save_image(sample_tensor, save_path, input_type='tensor')

        assert result is True
        assert save_path.exists()

    def test_save_creates_directory(self, sample_numpy, temp_dir):
        save_path = temp_dir / "nested" / "deep" / "output.jpg"
        result = save_image(sample_numpy, save_path, input_type='numpy')

        assert result is True
        assert save_path.exists()

    def test_save_none_image(self):
        with pytest.raises(ValueError, match="Empty image"):
            save_image(None, "output.jpg")

    def test_save_none_path(self, sample_numpy):
        with pytest.raises(ValueError, match="Empty path"):
            save_image(sample_numpy, None)


class TestShowImage:
    @patch('cv2.destroyAllWindows')
    @patch('cv2.waitKey')
    @patch('cv2.imshow')
    @patch('cv2.resizeWindow')
    @patch('cv2.namedWindow')
    @patch('feature_matching.image_utils.get_monitors')
    def test_show_numpy(self, mock_get_monitors, mock_named, mock_resize,
                        mock_imshow, mock_wait, mock_destroy, sample_numpy):
        mock_monitor = MagicMock()
        mock_monitor.width = 1920
        mock_monitor.height = 1080
        mock_get_monitors.return_value = [mock_monitor]
        mock_wait.return_value = 27

        show_image(sample_numpy, title="Test", input_type='numpy')

        mock_imshow.assert_called_once()
        mock_wait.assert_called_once_with(0)

    @patch('cv2.destroyAllWindows')
    @patch('cv2.waitKey')
    @patch('cv2.imshow')
    @patch('cv2.resizeWindow')
    @patch('cv2.namedWindow')
    @patch('feature_matching.image_utils.get_monitors')
    def test_show_tensor(self, mock_get_monitors, mock_named, mock_resize,
                         mock_imshow, mock_wait, mock_destroy, sample_tensor):
        mock_monitor = MagicMock()
        mock_monitor.width = 1920
        mock_monitor.height = 1080
        mock_get_monitors.return_value = [mock_monitor]
        mock_wait.return_value = 27

        show_image(sample_tensor, title="Test", input_type='tensor')

        mock_imshow.assert_called_once()
        mock_wait.assert_called_once_with(0)

    def test_show_none_image(self):
        with pytest.raises(ValueError, match="Empty image"):
            show_image(None)


class TestNumpyBGR:
    def test_numpy_input_rgb_to_bgr(self):
        rgb_img = np.zeros((1, 1, 3), dtype=np.uint8)
        rgb_img[0, 0] = [255, 0, 0]
        result = to_numpy_bgr(rgb_img, input_type='numpy')
        assert result[0, 0, 0] == 0
        assert result[0, 0, 1] == 0
        assert result[0, 0, 2] == 255
        assert result.shape == (1, 1, 3)

    def test_numpy_input_grayscale_to_bgr(self):
        gray_img = np.array([[128]], dtype=np.uint8)
        result = to_numpy_bgr(gray_img, input_type='numpy')
        assert result.shape == (1, 1, 3)
        assert result[0, 0, 0] == 128
        assert result[0, 0, 1] == 128
        assert result[0, 0, 2] == 128

    def test_tensor_input_to_bgr(self):
        tensor = torch.zeros(3, 1, 1)
        tensor[0] = 1.0
        tensor[1] = 0.0
        tensor[2] = 0.0
        result = to_numpy_bgr(tensor, input_type='tensor')
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 1, 3)
        assert result[0, 0, 0] == 0
        assert result[0, 0, 1] == 0
        assert result[0, 0, 2] == 255


class TestIntegration:
    def test_read_save_roundtrip(self, test_image_path, temp_dir):
        img = read_image(test_image_path, input_type='numpy')

        save_path = temp_dir / "roundtrip.png"
        save_image(img, save_path, input_type='numpy')

        img2 = read_image(save_path, input_type='numpy')

        assert img.shape == img2.shape
        np.testing.assert_array_equal(img, img2)

    def test_numpy_tensor_consistency(self, test_image_path):
        img_np = read_image(test_image_path, input_type='numpy')
        img_tensor = read_image(test_image_path, input_type='tensor')

        converter = Converter.create('image')
        img_back = converter.convert(img_tensor, 'tensor', 'opencv')

        assert img_np.shape == img_back.shape
        diff = np.abs(img_np.astype(np.float32) - img_back.astype(np.float32))
        assert diff.mean() < 5.0

    def test_tensor_numpy_consistency(self, test_image_path):
        img_np = read_image(test_image_path, input_type='numpy')
        img_tensor = read_image(test_image_path, input_type='tensor')

        converter = Converter.create('image')
        img_back = converter.convert(img_np, 'opencv', 'tensor')

        assert img_tensor.shape == img_back.shape
        diff = torch.abs(img_tensor - img_back)
        assert diff.mean().item() < 5.0
