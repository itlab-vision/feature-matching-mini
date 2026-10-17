import pytest
import numpy as np
import cv2 as cv
from unittest.mock import MagicMock, patch
from src.backend.opencv.io_adapter.disk_io_adapter import DiskOpenCVIOAdapter


@pytest.fixture
def mock_model():
    return MagicMock()


@pytest.fixture
def adapter(mock_model):
    logger = MagicMock()
    config = {}
    return DiskOpenCVIOAdapter(
        model_name="disk_opencv",
        model_wrapper=mock_model,
        config=config,
        logger=logger
    )


def test_disk_preprocess(adapter, mock_model):
    fake_image = np.zeros((800, 1200, 3), dtype=np.uint8)
    inputs = {'image': fake_image}

    with patch('cv2.dnn.blobFromImage') as mock_blob:
        mock_blob.return_value = "fake_disk_blob"

        adapter.preprocess(inputs)

        assert adapter.orig_h == 800
        assert adapter.orig_w == 1200

        mock_blob.assert_called_once_with(
            fake_image,
            scalefactor=1.0 / 255.0,
            size=(1024, 1024),
            swapRB=True,
            crop=False
        )

        mock_model.setInput.assert_called_once_with("fake_disk_blob")


def test_disk_postprocess(adapter):
    adapter.orig_h = 2048
    adapter.orig_w = 2048

    fake_keypoints = np.array([[512.0, 512.0], [0.0, 0.0]], dtype=np.float32)
    fake_descriptors = np.random.rand(1, 2, 128).astype(np.float32)
    fake_scores = np.array([[0.9, 0.8]], dtype=np.float32)

    outputs = [fake_keypoints, fake_descriptors, fake_scores]

    result = adapter.postprocess(outputs)

    assert isinstance(result, dict)
    assert 'keypoints' in result
    assert 'descriptors' in result
    assert 'scores' in result

    assert result['descriptors'].shape == (2, 128)
    assert result['scores'].shape == (2,)

    keypoints = result['keypoints']
    assert len(keypoints) == 2
    assert isinstance(keypoints[0], cv.KeyPoint)

    assert keypoints[0].pt == (1024.0, 1024.0)
    assert keypoints[1].pt == (0.0, 0.0)
