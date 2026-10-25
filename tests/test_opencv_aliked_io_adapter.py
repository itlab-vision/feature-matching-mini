import pytest
import numpy as np
import cv2 as cv
from unittest.mock import MagicMock, patch
from src.backend.opencv.io_adapter.aliked_io_adapter import AlikedOpenCVIOAdapter


@pytest.fixture
def mock_model():
    return MagicMock()


@pytest.fixture
def adapter(mock_model):
    logger = MagicMock()
    config = {}
    return AlikedOpenCVIOAdapter(
        model_name="aliked_opencv",
        model_wrapper=mock_model,
        config=config,
        logger=logger
    )


def test_preprocess(adapter, mock_model):
    fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
    inputs = {'image': fake_image}

    with patch('cv2.dnn.blobFromImage') as mock_blob:
        mock_blob.return_value = "fake_blob_tensor"

        adapter.preprocess(inputs)

        assert adapter.orig_h == 480
        assert adapter.orig_w == 640

        mock_blob.assert_called_once_with(
            fake_image,
            scalefactor=1.0 / 255.0,
            size=(640, 640),
            swapRB=True,
            crop=False
        )

        mock_model.setInput.assert_called_once_with("fake_blob_tensor")


def test_postprocess(adapter):
    adapter.orig_h = 1000
    adapter.orig_w = 2000

    fake_keypoints = np.array([[[0.0, 0.0], [-1.0, -1.0]]], dtype=np.float32)
    fake_descriptors = np.random.rand(2, 128).astype(np.float32)
    fake_scores = np.array([0.95, 0.88], dtype=np.float32)

    outputs = [fake_keypoints, fake_descriptors, fake_scores]

    result = adapter.postprocess(outputs)

    assert isinstance(result, dict)
    assert 'keypoints' in result
    assert 'descriptors' in result
    assert 'scores' in result

    np.testing.assert_array_equal(result['descriptors'], fake_descriptors)
    np.testing.assert_array_equal(result['scores'], fake_scores)

    keypoints = result['keypoints']
    assert len(keypoints) == 2
    assert isinstance(keypoints[0], cv.KeyPoint)

    assert keypoints[0].pt == (1000.0, 500.0)
    assert keypoints[1].pt == (0.0, 0.0)
