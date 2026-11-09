from unittest.mock import MagicMock, patch

import pytest
import torch

from src.backend.executorch.model_wrapper.xfeat import XFeatExecuTorchModelWrapper
from src.backend.model_wrapper_base import ModelWrapper


@pytest.fixture
def wrapper():
    wrapper = XFeatExecuTorchModelWrapper("xfeat_executorch", logger=MagicMock())
    wrapper._method = MagicMock()
    return wrapper


class TestXFeatExecuTorchModelWrapperRegistration:
    def test_registered_in_factory(self):
        assert ModelWrapper._METHODS["xfeat_executorch"] is XFeatExecuTorchModelWrapper


class TestXFeatExecuTorchModelWrapperLoad:
    @patch("src.backend.executorch.model_wrapper.xfeat.ExecuTorchModelWrapper.load")
    def test_load_delegates_to_base_wrapper(self, mock_load):
        expected = {"model": MagicMock()}
        mock_load.return_value = expected
        wrapper = XFeatExecuTorchModelWrapper("xfeat_executorch")
        assert wrapper.load() is expected


class TestXFeatExecuTorchModelWrapperCall:
    def test_call_requires_loaded_model(self):
        with pytest.raises(RuntimeError, match="load"):
            XFeatExecuTorchModelWrapper("xfeat_executorch").call(torch.zeros(1))
