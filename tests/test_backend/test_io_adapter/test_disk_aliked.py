import cv2 as cv
import numpy as np
import pytest

from src.backend.io_adapter import ALIKEDIOAdapter, DISKIOAdapter
from src.backend.io_adapter_base import IOAdapter


@pytest.mark.parametrize('name, adapter_class, point', [
    ('disk', DISKIOAdapter, [512, 512]),
    ('aliked', ALIKEDIOAdapter, [0, 0]),
])
@pytest.mark.parametrize('backend', ['opencv', 'onnx'])
def test_disk_and_aliked_use_same_numpy_adapter(name, adapter_class, point, backend):
    adapter = IOAdapter.create(backend, name)
    assert isinstance(adapter, adapter_class)
    image = np.zeros((512, 512), dtype=np.uint8)
    inputs = adapter.preprocess({'image': image, 'extra': 7})
    assert inputs['image'].shape == (512, 512, 3)
    assert inputs['extra'] == 7

    outputs = adapter.postprocess((np.asarray([point], dtype=np.float32),
                                   np.ones((1, 1, 128), dtype=np.float32),
                                   np.array([[0.9]], dtype=np.float32)))
    assert isinstance(outputs['keypoints'][0], cv.KeyPoint)
    assert outputs['keypoints'][0].pt == (256.0, 256.0)
    assert outputs['descriptors'].shape == (1, 128)
    np.testing.assert_allclose(outputs['scores'], [0.9])


@pytest.mark.parametrize('name', ['disk', 'aliked'])
def test_disk_and_aliked_reject_wrong_image_channels(name):
    with pytest.raises(ValueError):
        IOAdapter.create('onnx', name).preprocess({'image': np.zeros((4, 4, 4))})
