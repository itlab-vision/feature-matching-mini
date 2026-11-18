from unittest.mock import MagicMock, patch

import pytest
import torch

from src.backend.executorch.model_wrapper.d2net import D2NetExecuTorchModelWrapper
from src.backend.model_wrapper_base import ModelWrapper


@pytest.fixture
def wrapper():
    wrapper = D2NetExecuTorchModelWrapper("d2net_executorch", logger=MagicMock())
    wrapper._method = MagicMock()
    return wrapper


class TestD2NetExecuTorchModelWrapperRegistration:
    def test_registered_in_factory(self):
        assert ModelWrapper._METHODS["d2net_executorch"] is D2NetExecuTorchModelWrapper


class TestD2NetExecuTorchModelWrapperLoad:
    @patch("src.backend.executorch.model_wrapper.d2net.ExecuTorchModelWrapper.load")
    def test_load_delegates_to_base_wrapper(self, mock_load):
        expected = {"model": MagicMock()}
        mock_load.return_value = expected
        wrapper = D2NetExecuTorchModelWrapper("d2net_executorch")
        assert wrapper.load() is expected


class TestD2NetExecuTorchModelWrapperCall:
    def test_call_requires_loaded_model(self):
        with pytest.raises(RuntimeError, match="load"):
            D2NetExecuTorchModelWrapper("d2net_executorch").call(torch.zeros(1))
