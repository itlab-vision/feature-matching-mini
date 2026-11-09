import cv2 as cv
import numpy as np

from src.backend.io_adapter_base import IOAdapter
from src.backend.io_adapter import SuperPointLightGlueIOAdapter


def test_superpoint_lightglue_shared_adapter():
    adapter = IOAdapter.create('executorch', 'superpoint_lightglue')
    assert isinstance(adapter, SuperPointLightGlueIOAdapter)
    inputs = adapter.preprocess({'image': np.zeros((8, 16, 3), dtype=np.uint8)})
    assert isinstance(inputs['image'], np.ndarray)
    outputs = adapter.postprocess({'keypoints': np.array([[4, 2]], dtype=np.float32),
                                   'descriptors': np.ones((1, 4), dtype=np.float32),
                                   'scores': np.array([0.8], dtype=np.float32)})
    assert isinstance(outputs['keypoints'][0], cv.KeyPoint)
    assert outputs['descriptors'].shape == (1, 4)
