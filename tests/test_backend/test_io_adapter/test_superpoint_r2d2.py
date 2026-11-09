import cv2 as cv
import numpy as np
import pytest

from src.backend.io_adapter import R2D2IOAdapter, SuperPointIOAdapter
from src.backend.io_adapter_base import IOAdapter


@pytest.mark.parametrize('name, adapter_class', [
    ('superpoint', SuperPointIOAdapter),
    ('r2d2', R2D2IOAdapter),
])
def test_feature_adapter_preserves_numpy_boundary(name, adapter_class):
    adapter = IOAdapter.create('torch', name)
    assert isinstance(adapter, adapter_class)
    inputs = adapter.preprocess({'image': np.zeros((6, 8), dtype=np.uint8)})
    assert inputs['image'].shape == (6, 8, 3)
    assert isinstance(inputs['image'], np.ndarray)

    outputs = adapter.postprocess({'keypoints': np.array([[2, 3]], dtype=np.float32),
                                   'descriptors': np.ones((1, 16), dtype=np.float32),
                                   'scores': np.array([0.8], dtype=np.float32)})
    assert isinstance(outputs['keypoints'][0], cv.KeyPoint)
    assert outputs['keypoints'][0].pt == (2.0, 3.0)
    assert outputs['descriptors'].shape == (1, 16)
