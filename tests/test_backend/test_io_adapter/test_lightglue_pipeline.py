import cv2 as cv
import numpy as np
import pytest

from src.backend.io_adapter import (ALIKEDLightGlueIOAdapter, DISKLightGlueIOAdapter,
                                    DoGHardNetLightGlueIOAdapter, SIFTLightGlueIOAdapter,
                                    SuperPointLightGlueIOAdapter)
from src.backend.io_adapter_base import IOAdapter


@pytest.mark.parametrize('name, adapter_class', [
    ('superpoint_lightglue', SuperPointLightGlueIOAdapter),
    ('disk_lightglue', DISKLightGlueIOAdapter),
    ('sift_lightglue', SIFTLightGlueIOAdapter),
    ('aliked_lightglue', ALIKEDLightGlueIOAdapter),
    ('doghardnet_lightglue', DoGHardNetLightGlueIOAdapter),
])
def test_lightglue_extractor_adapter_uses_numpy(name, adapter_class):
    adapter = IOAdapter.create('torch', name)
    assert isinstance(adapter, adapter_class)
    image = np.zeros((6, 8), dtype=np.uint8)
    inputs = adapter.preprocess({'image': image})
    assert inputs['image'].shape == (6, 8, 3)

    outputs = adapter.postprocess({'keypoints': np.array([[2, 3]], dtype=np.float32),
                                   'descriptors': np.ones((1, 8), dtype=np.float32),
                                   'scores': np.array([0.8], dtype=np.float32)})
    assert isinstance(outputs['keypoints'][0], cv.KeyPoint)
    assert outputs['keypoints'][0].pt == (2.0, 3.0)
    assert outputs['descriptors'].shape == (1, 8)
