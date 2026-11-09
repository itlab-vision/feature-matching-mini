from unittest.mock import MagicMock, patch

import pytest
import torch

from src.backend.executorch.model_wrapper.disk_lightglue import (
    DiskLightGlueExecuTorchModelWrapper,
)
from src.backend.model_wrapper_base import ModelWrapper


@pytest.fixture
def wrapper():
    wrapper = DiskLightGlueExecuTorchModelWrapper("disk_lightglue_executorch", logger=MagicMock())
    wrapper._method = MagicMock()
    return wrapper


class TestDiskLightGlueExecuTorchModelWrapperRegistration:
    def test_registered_in_factory(self):
        assert (
            ModelWrapper._METHODS["disk_lightglue_executorch"]
            is DiskLightGlueExecuTorchModelWrapper
        )


class TestDiskLightGlueExecuTorchModelWrapperLoad:
    @patch("src.backend.executorch.model_wrapper.disk_lightglue.ExecuTorchModelWrapper.load")
    def test_load_delegates_to_base_wrapper(self, mock_load):
        expected = {"model": MagicMock()}
        mock_load.return_value = expected
        wrapper = DiskLightGlueExecuTorchModelWrapper("disk_lightglue_executorch")
        assert wrapper.load() is expected


class TestDiskLightGlueExecuTorchModelWrapperCall:

    def test_call_requires_loaded_model(self):
        with pytest.raises(RuntimeError, match="load"):
            DiskLightGlueExecuTorchModelWrapper("disk_lightglue_executorch").call(torch.zeros(1))
