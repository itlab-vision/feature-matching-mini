import cv2 as cv
import numpy as np
import pytest

from src.backend.io_adapter import D2NetIOAdapter
from src.backend.io_adapter_base import IOAdapter


def test_d2net_shared_adapter_preprocess_and_postprocess():
    adapter = IOAdapter.create('executorch', 'd2net')
    assert isinstance(adapter, D2NetIOAdapter)
    inputs = adapter.preprocess({'image': np.zeros((4, 6), dtype=np.uint8)})
    assert inputs['image'].shape == (4, 6, 3)
    assert isinstance(inputs['image'], np.ndarray)
    outputs = adapter.postprocess({'keypoints': np.array([[2, 1]], dtype=np.float32),
                                   'descriptors': np.ones((1, 128), dtype=np.float32),
                                   'scores': np.array([0.9], dtype=np.float32)})
    assert isinstance(outputs['keypoints'][0], cv.KeyPoint)
    assert outputs['descriptors'].shape == (1, 128)


@pytest.mark.parametrize('inputs', [{}, None, []])
def test_d2net_requires_image(inputs):
    with pytest.raises(ValueError):
        IOAdapter.create('executorch', 'd2net').preprocess(inputs)
