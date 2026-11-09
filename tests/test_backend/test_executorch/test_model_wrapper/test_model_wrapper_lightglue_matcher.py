from unittest.mock import MagicMock, patch

import pytest
import torch

from src.backend.executorch.model_wrapper.lightglue_matcher import (
    LightGlueExecuTorchModelWrapper,
)
from src.backend.model_wrapper_base import ModelWrapper


@pytest.fixture
def wrapper():
    wrapper = LightGlueExecuTorchModelWrapper("lightglue_executorch", logger=MagicMock())
    wrapper._method = MagicMock()
    return wrapper


@pytest.fixture
def inputs():
    return {
        "keypoints0": torch.zeros(1, 2, 2),
        "keypoints1": torch.zeros(1, 2, 2),
        "descriptors0": torch.zeros(1, 2, 4),
        "descriptors1": torch.zeros(1, 2, 4),
    }


class TestLightGlueExecuTorchModelWrapperRegistration:
    @pytest.mark.parametrize(
        "name",
        ["lightglue_executorch", "lightglue_disk_executorch", "lightglue_superpoint_executorch"],
    )
    def test_registered_aliases(self, name):
        assert ModelWrapper._METHODS[name] is LightGlueExecuTorchModelWrapper


class TestLightGlueExecuTorchModelWrapperLoad:
    @patch("src.backend.executorch.model_wrapper.lightglue_matcher.ExecuTorchModelWrapper.load")
    def test_load_delegates_to_base_wrapper(self, mock_load):
        expected = {"model": MagicMock()}
        mock_load.return_value = expected
        wrapper = LightGlueExecuTorchModelWrapper("lightglue_executorch")
        assert wrapper.load() is expected


class TestLightGlueExecuTorchModelWrapperCall:

    def test_call_requires_loaded_model(self, inputs):
        wrapper = LightGlueExecuTorchModelWrapper("lightglue_executorch")
        with pytest.raises(RuntimeError, match="load"):
            wrapper.call(inputs)
