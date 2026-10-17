import pytest
import cv2 as cv
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil

from src.image_utils import read_image, save_image, show_image


@pytest.fixture
def temp_dir():
    dir_path = tempfile.mkdtemp()
    yield Path(dir_path)
    shutil.rmtree(dir_path)


@pytest.fixture
def test_image_path(temp_dir):
    img_path = temp_dir / "test_image.png"
    test_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    cv.imwrite(str(img_path), test_img)
    return img_path


@pytest.fixture
def sample_numpy():
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)


class TestReadImage:

    def test_read_image(self, test_image_path):
        img = read_image(test_image_path)

        assert isinstance(img, np.ndarray)
        assert img.shape == (100, 100, 3)
        assert img.dtype == np.uint8

    def test_read_none_path(self):
        with pytest.raises(ValueError, match="Empty path to the image"):
            read_image(None)

    def test_read_nonexistent_path(self, temp_dir):
        nonexistent_path = temp_dir / "nonexistent.png"

        with pytest.raises(ValueError, match="Incorrect path to the image"):
            read_image(nonexistent_path)

    def test_read_invalid_image(self, temp_dir):
        invalid_path = temp_dir / "invalid.png"
        invalid_path.write_text("not an image")

        with pytest.raises(ValueError, match="Failed to read image from"):
            read_image(invalid_path)


class TestSaveImage:
    def test_save_numpy(self, sample_numpy, temp_dir):
        save_path = temp_dir / "output.png"
        result = save_image(sample_numpy, save_path)

        assert result is True
        assert save_path.exists()

        saved_img = cv.imread(str(save_path))
        assert saved_img is not None
        assert saved_img.shape == sample_numpy.shape
        np.testing.assert_array_equal(saved_img, sample_numpy)

    def test_save_creates_directory(self, sample_numpy, temp_dir):
        save_path = temp_dir / "nested" / "deep" / "output.png"
        result = save_image(sample_numpy, save_path)

        assert result is True
        assert save_path.exists()
        assert save_path.parent.exists()

    def test_save_none_image(self, temp_dir):
        save_path = temp_dir / "output.png"

        with pytest.raises(ValueError, match="Empty image"):
            save_image(None, save_path)

    def test_save_none_path(self, sample_numpy):
        with pytest.raises(ValueError, match="Empty path to save"):
            save_image(sample_numpy, None)


class TestShowImage:
    @patch("src.image_utils.cv.destroyAllWindows")
    @patch("src.image_utils.cv.waitKey")
    @patch("src.image_utils.cv.imshow")
    @patch("src.image_utils.cv.resizeWindow")
    @patch("src.image_utils.cv.namedWindow")
    @patch("src.image_utils.get_monitors")
    def test_show_numpy(self, mock_get_monitors, mock_named, mock_resize, mock_imshow, mock_wait, mock_destroy,
                        sample_numpy):
        mock_monitor = MagicMock()
        mock_monitor.width = 1920
        mock_monitor.height = 1080
        mock_get_monitors.return_value = [mock_monitor]
        mock_wait.return_value = 27

        show_image(sample_numpy, title="Test")

        mock_named.assert_called_once_with("Test", cv.WINDOW_NORMAL)
        mock_resize.assert_called_once_with("Test", 100, 100)
        mock_imshow.assert_called_once_with("Test", sample_numpy)
        mock_wait.assert_called_once_with(0)
        mock_destroy.assert_called_once()

    @patch("src.image_utils.cv.destroyAllWindows")
    @patch("src.image_utils.cv.waitKey")
    @patch("src.image_utils.cv.imshow")
    @patch("src.image_utils.cv.resizeWindow")
    @patch("src.image_utils.cv.namedWindow")
    @patch("src.image_utils.get_monitors")
    def test_show_resizes_to_monitor(self, mock_get_monitors, mock_named, mock_resize, mock_imshow, mock_wait,
                                     mock_destroy, temp_dir):
        mock_monitor = MagicMock()
        mock_monitor.width = 800
        mock_monitor.height = 600
        mock_get_monitors.return_value = [mock_monitor]

        large_img = np.zeros((1000, 1200, 3), dtype=np.uint8)
        show_image(large_img, title="Large image")
        mock_resize.assert_called_once_with("Large image", 800, 600)

    def test_show_none_image(self):
        with pytest.raises(ValueError, match="Empty image to show"):
            show_image(None)


class TestIntegration:
    def test_read_save_roundtrip(self, test_image_path, temp_dir):
        img = read_image(test_image_path)
        save_path = temp_dir / "roundtrip.png"
        result = save_image(img, save_path)
        assert result is True
        img2 = read_image(save_path)

        assert img.shape == img2.shape
        np.testing.assert_array_equal(img, img2)

    def test_read_save_preserves_dtype(self, test_image_path, temp_dir):
        img = read_image(test_image_path)
        save_path = temp_dir / "copy.png"
        save_image(img, save_path)
        img2 = read_image(save_path)

        assert img.dtype == np.uint8
        assert img2.dtype == np.uint8
