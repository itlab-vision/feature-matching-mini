import cv2 as cv
import numpy as np
import pytest

from src.backend.io_adapter import SuperGlueIOAdapter
from src.backend.io_adapter_base import IOAdapter


def test_superglue_converts_feature_pairs_and_matches():
    adapter = IOAdapter.create('torch', 'superglue')
    assert isinstance(adapter, SuperGlueIOAdapter)
    features = {'keypoints': [cv.KeyPoint(4, 6, 1)],
                'descriptors': np.ones((1, 8), dtype=np.float32),
                'scores': np.array([0.8], dtype=np.float32)}
    inputs = adapter.preprocess({'features0': features, 'features1': features})
    np.testing.assert_array_equal(inputs['features0']['keypoints'], [[4, 6]])
    assert isinstance(inputs['features0']['descriptors'], np.ndarray)

    outputs = adapter.postprocess({'matches': np.array([[0, 0]]), 'scores': [0.7]})
    assert isinstance(outputs['matches'][0], cv.DMatch)
    np.testing.assert_allclose(outputs['scores'], [0.7])


def test_superglue_requires_both_feature_sets():
    with pytest.raises(ValueError):
        IOAdapter.create('torch', 'superglue').preprocess({'features0': {}})
