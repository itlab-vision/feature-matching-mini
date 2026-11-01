import pytest
import numpy as np
import cv2 as cv
from unittest.mock import MagicMock
from src.backend.onnx.io_adapter.lightglue_io_adapter import LightGlueOnnxIOAdapter


@pytest.fixture
def mock_model():
    return MagicMock()


@pytest.fixture
def adapter(mock_model):
    logger = MagicMock()
    config = {'score_threshold': 0.1}
    adapter_instance = LightGlueOnnxIOAdapter(
        model_name="lightglue_onnx",
        model_wrapper=mock_model,
        config=config,
        logger=logger
    )
    return adapter_instance


def test_normalize_keypoints_disk(adapter):
    keypoints = np.array([[100.0, 200.0], [500.0, 400.0]], dtype=np.float32)
    img_shape = (1000, 2000, 3)
    height, width = img_shape[:2]

    normalized = adapter._normalize_keypoints(keypoints, height, width, extractor_name="disk")

    expected = np.array([
        [100.0 / (width - 1), 200.0 / (height - 1)],
        [500.0 / (width - 1), 400.0 / (height - 1)]
    ], dtype=np.float32)

    np.testing.assert_allclose(normalized, expected, rtol=1e-5, atol=1e-5)


def test_normalize_keypoints_aliked(adapter):
    keypoints = np.array([[0.0, 0.0], [2000.0, 1000.0]], dtype=np.float32)
    img_shape = (1000, 2000, 3)
    height, width = img_shape[:2]

    normalized = adapter._normalize_keypoints(keypoints, height, width, extractor_name="aliked")

    expected = np.array([
        [2.0 * 0.0 / width - 1.0, 2.0 * 0.0 / height - 1.0],
        [2.0 * 2000.0 / width - 1.0, 2.0 * 1000.0 / height - 1.0]
    ], dtype=np.float32)

    np.testing.assert_allclose(normalized, expected, rtol=1e-5, atol=1e-5)


def test_preprocess(adapter, mock_model):
    inputs = {
        'features0': {
            'keypoints': np.array([[0.0, 0.0]], dtype=np.float32),
            'descriptors': np.random.rand(1, 128).astype(np.float32),
            'height': 100,
            'width': 100,
            'extractor_name': 'disk'
        },
        'features1': {
            'keypoints': np.array([[10.0, 10.0]], dtype=np.float32),
            'descriptors': np.random.rand(1, 128).astype(np.float32),
            'height': 100,
            'width': 100,
            'extractor_name': 'disk'
        }
    }

    processed = adapter.preprocess(inputs)

    assert isinstance(processed, dict)
    assert 'kpts0' in processed
    assert 'kpts1' in processed
    assert 'desc0' in processed
    assert 'desc1' in processed

    mock_model.assert_not_called()


def test_postprocess(adapter):
    adapter.scoreThreshold = 0.5

    fake_matches = np.array([[0, 1], [2, 3]], dtype=np.int64)
    fake_scores = np.array([0.9, 0.2], dtype=np.float32)

    outputs = [fake_matches, fake_scores]

    result = adapter.postprocess(outputs)

    assert isinstance(result, dict)
    assert 'matches' in result
    assert 'scores' in result

    dmatches = result['matches']
    assert len(dmatches) == 1
    assert isinstance(dmatches[0], cv.DMatch)

    assert dmatches[0].queryIdx == 0
    assert dmatches[0].trainIdx == 1

    assert pytest.approx(dmatches[0].distance, 0.01) == 0.1
