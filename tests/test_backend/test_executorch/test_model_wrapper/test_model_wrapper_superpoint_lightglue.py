from unittest.mock import MagicMock, patch

import pytest
import torch

from src.backend.executorch.model_wrapper.superpoint_lightglue import (
    SuperPointLightGlueExecuTorchModelWrapper,
)
from src.backend.model_wrapper_base import ModelWrapper


@pytest.fixture
def wrapper():
    wrapper = SuperPointLightGlueExecuTorchModelWrapper(
        "superpoint_lightglue_executorch", logger=MagicMock()
    )
    wrapper._method = MagicMock()
    return wrapper


class TestSuperPointLightGlueExecuTorchModelWrapperRegistration:
    def test_registered_in_factory(self):
        assert (
            ModelWrapper._METHODS["superpoint_lightglue_executorch"]
            is SuperPointLightGlueExecuTorchModelWrapper
        )


class TestSuperPointLightGlueExecuTorchModelWrapperLoad:
    @patch("src.backend.executorch.model_wrapper.superpoint_lightglue.ExecuTorchModelWrapper.load")
    def test_load_delegates_to_base_wrapper(self, mock_load):
        expected = {"model": MagicMock()}
        mock_load.return_value = expected
        wrapper = SuperPointLightGlueExecuTorchModelWrapper("superpoint_lightglue_executorch")
        assert wrapper.load() is expected


class TestSuperPointLightGlueExecuTorchModelWrapperCall:

    def test_call_requires_loaded_model(self):
        wrapper = SuperPointLightGlueExecuTorchModelWrapper("superpoint_lightglue_executorch")
        with pytest.raises(RuntimeError, match="load"):
            wrapper.call(torch.zeros(1))
