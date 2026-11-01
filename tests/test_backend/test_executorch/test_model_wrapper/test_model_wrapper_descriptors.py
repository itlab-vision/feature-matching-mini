from unittest.mock import MagicMock

import pytest
import torch

from src.backend.executorch.model_wrapper.descriptors import (
    PatchDescriptorExecuTorchModelWrapper,
)
from src.backend.model_wrapper_base import ModelWrapper


@pytest.fixture
def wrapper():
    wrapper = PatchDescriptorExecuTorchModelWrapper(
        "hardnet_executorch", config={"descriptor_dim": 4}, logger=MagicMock()
    )
    wrapper._method = MagicMock()
    return wrapper


class TestPatchDescriptorExecuTorchModelWrapperRegistration:
    @pytest.mark.parametrize("name", ["hardnet_executorch", "tfeat_executorch"])
    def test_registered_aliases(self, name):
        assert ModelWrapper._METHODS[name] is PatchDescriptorExecuTorchModelWrapper


class TestPatchDescriptorExecuTorchModelWrapperCall:
    def test_call_runs_each_patch_separately_and_concatenates_outputs(self, wrapper):
        patches = torch.arange(128, dtype=torch.float32).reshape(2, 1, 8, 8)
        wrapper._method.execute.side_effect = [
            (torch.tensor([[1.0, 2.0, 3.0, 4.0]]),),
            torch.tensor([[5.0, 6.0, 7.0, 8.0]]),
        ]

        result = wrapper.call({"patches": patches})

        assert wrapper._method.execute.call_count == 2
        assert all(call.args[0][0].shape == (1, 1, 8, 8) for call in wrapper._method.execute.call_args_list)
        torch.testing.assert_close(
            result,
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
        )

    def test_call_returns_configured_empty_descriptor_shape(self, wrapper):
        result = wrapper.call(torch.empty(0, 1, 8, 8))
        assert result.shape == (0, 4)
        assert result.dtype == torch.float32
        wrapper._method.execute.assert_not_called()

    def test_call_requires_loaded_model(self):
        wrapper = PatchDescriptorExecuTorchModelWrapper("hardnet_executorch")
        with pytest.raises(RuntimeError, match="load"):
            wrapper.call(torch.zeros(1, 1, 8, 8))

    @pytest.mark.parametrize("patches", ["patches", torch.zeros(1, 8, 8)])
    def test_call_rejects_invalid_patch_batch(self, wrapper, patches):
        with pytest.raises(TypeError, match="N, C, H, W"):
            wrapper.call(patches)

    def test_call_logs_and_reraises_runtime_error(self, wrapper):
        wrapper._method.execute.side_effect = RuntimeError("inference failed")
        with pytest.raises(RuntimeError, match="inference failed"):
            wrapper.call(torch.zeros(1, 1, 8, 8))
        wrapper._logger.warning.assert_called_once_with(
            "hardnet_executorch ExecuTorch inference error: inference failed"
        )
