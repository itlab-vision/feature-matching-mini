import cv2 as cv
import numpy as np
import pytest

from src.backend.io_adapter import LoFTRIOAdapter, RoMaIOAdapter
from src.backend.io_adapter_base import IOAdapter


@pytest.mark.parametrize('name, adapter_class', [('loftr', LoFTRIOAdapter), ('roma', RoMaIOAdapter)])
def test_dense_matcher_adapter_converts_pairs(name, adapter_class):
    adapter = IOAdapter.create('torch', name)
    assert isinstance(adapter, adapter_class)
    images = adapter.preprocess({'image0': np.zeros((4, 6), dtype=np.uint8),
                                 'image1': np.ones((4, 6), dtype=np.uint8)})
    assert all(isinstance(images[key], np.ndarray) for key in ('image0', 'image1'))

    outputs = adapter.postprocess({'keypoints0': np.array([[1, 2]], dtype=np.float32),
                                   'keypoints1': np.array([[3, 4]], dtype=np.float32),
                                   'matches': np.array([[0, 0]], dtype=np.int64),
                                   'scores': np.array([0.9], dtype=np.float32)})
    assert outputs['keypoints0'][0].pt == (1.0, 2.0)
    assert outputs['keypoints1'][0].pt == (3.0, 4.0)
    assert isinstance(outputs['matches'][0], cv.DMatch)
    np.testing.assert_allclose(outputs['scores'], [0.9])


@pytest.mark.parametrize('name', ['loftr', 'roma'])
def test_dense_matcher_requires_two_images(name):
    with pytest.raises(ValueError):
        IOAdapter.create('torch', name).preprocess({'image0': np.zeros((4, 6))})
