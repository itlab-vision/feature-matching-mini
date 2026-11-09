from unittest.mock import MagicMock

import cv2 as cv
import numpy as np
import pytest
import torch

from src.backend.executorch.model_wrapper.tfeat import TFeatExecuTorchModelWrapper
from src.backend.executorch.model_wrapper.hardnet import HardNetExecuTorchModelWrapper
from src.backend.torch.model_wrapper.tfeat import TFeatTorchModelWrapper
from src.backend.model_wrapper_base import ModelWrapper


@pytest.mark.parametrize('name, wrapper_class', [
    ('tfeat', TFeatExecuTorchModelWrapper),
    ('hardnet', HardNetExecuTorchModelWrapper),
])
def test_patch_wrapper_registration_and_inference(name, wrapper_class):
    assert ModelWrapper._METHODS[f'{name}_executorch'] is wrapper_class
    wrapper = wrapper_class(name, config={'patch_size': 8})
    wrapper._method = MagicMock()
    wrapper._method.execute.return_value = (torch.ones((1, 128)),)
    inputs = {'image': np.zeros((16, 16), dtype=np.uint8),
              'keypoints': np.asarray([cv.KeyPoint(8, 8, 3)], dtype=object),
              'scores': np.asarray([0.8], dtype=np.float32)}

    result = wrapper.call(inputs)

    wrapper._method.execute.assert_called_once()
    assert result['descriptors'].shape == (1, 128)
    np.testing.assert_allclose(result['scores'], [0.8])


def test_empty_patch_batch_skips_executorch():
    wrapper = HardNetExecuTorchModelWrapper('hardnet')
    wrapper._method = MagicMock()
    result = wrapper.call({'image': np.zeros((16, 16), dtype=np.uint8),
                           'keypoints': np.empty(0, dtype=object),
                           'scores': np.empty(0, dtype=np.float32)})
    assert result['descriptors'].shape == (0, 128)
    wrapper._method.execute.assert_not_called()


def test_tfeat_torch_and_executorch_use_identical_patches():
    image = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    inputs = {'image': image,
              'keypoints': np.asarray([cv.KeyPoint(32, 32, 8), cv.KeyPoint(20, 24, 12)], dtype=object),
              'scores': np.asarray([1.0, 0.5], dtype=np.float32)}
    torch_wrapper = TFeatTorchModelWrapper('tfeat', config={'device': 'cpu'})
    executorch_wrapper = TFeatExecuTorchModelWrapper('tfeat')

    torch_patches = torch_wrapper.preprocess(inputs)[0]
    executorch_patches = executorch_wrapper.preprocess(inputs)[0]

    torch.testing.assert_close(torch_patches.cpu(), executorch_patches, rtol=0, atol=0)
