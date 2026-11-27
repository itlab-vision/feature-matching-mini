import cv2 as cv
import numpy as np
import pytest

import src.backend.io_adapter  # noqa: F401
from src.backend.io_adapter_base import IOAdapter


@pytest.mark.parametrize('name', ['tfeat', 'hardnet'])
@pytest.mark.parametrize('backend', ['torch', 'executorch'])
def test_patch_descriptor_uses_shared_numpy_adapter(name, backend):
    adapter = IOAdapter.create(backend, name)
    point = cv.KeyPoint(8, 8, 3)
    inputs = adapter.preprocess({'image': np.zeros((16, 16), dtype=np.uint8),
                                 'keypoints': [point], 'scores': [0.7]})
    assert isinstance(inputs['image'], np.ndarray)
    result = adapter.postprocess({'keypoints': inputs['keypoints'],
                                  'descriptors': np.ones((1, 128)), 'scores': inputs['scores']})
    assert result['keypoints'][0] is point
    assert result['descriptors'].shape == (1, 128)
    np.testing.assert_allclose(result['scores'], [0.7])
