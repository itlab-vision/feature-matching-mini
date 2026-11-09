import cv2 as cv
import numpy as np
import pytest

from src.backend.io_adapter import LightGlueIOAdapter
from src.backend.io_adapter_base import IOAdapter


@pytest.mark.parametrize('name', ['lightglue', 'lightglue_disk', 'lightglue_superpoint'])
def test_lightglue_shared_adapter_uses_numpy_features(name):
    adapter = IOAdapter.create('executorch', name)
    assert isinstance(adapter, LightGlueIOAdapter)
    features = {'keypoints': [cv.KeyPoint(4, 6, 3)],
                'descriptors': np.ones((1, 128), dtype=np.float32),
                'scores': [0.8], 'height': 16, 'width': 16}
    inputs = adapter.preprocess({'features0': features, 'features1': features})
    assert inputs['features0']['keypoints'].shape == (1, 2)
    assert isinstance(inputs['features0']['descriptors'], np.ndarray)
    outputs = adapter.postprocess({'matches': np.array([[0, 0]]), 'scores': [0.7]})
    assert isinstance(outputs['matches'][0], cv.DMatch)
    np.testing.assert_allclose(outputs['scores'], [0.7])
