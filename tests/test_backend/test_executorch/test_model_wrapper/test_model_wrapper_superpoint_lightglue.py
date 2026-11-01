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
    def test_call_executes_contiguous_cpu_image(self, wrapper):
        image = torch.zeros(1, 3, 4, 4).transpose(2, 3)
        expected = tuple(torch.zeros(1) for _ in range(3))
        wrapper._method.execute.return_value = expected

        result = wrapper.call({"image": image})

        executed = wrapper._method.execute.call_args.args[0][0]
        assert executed.device.type == "cpu"
        assert executed.is_contiguous()
        assert result is expected

    def test_call_requires_loaded_model(self):
        wrapper = SuperPointLightGlueExecuTorchModelWrapper("superpoint_lightglue_executorch")
        with pytest.raises(RuntimeError, match="load"):
            wrapper.call(torch.zeros(1))

    def test_call_rejects_non_tensor(self, wrapper):
        with pytest.raises(TypeError, match="torch.Tensor"):
            wrapper.call({"image": None})

    def test_call_logs_and_reraises_runtime_error(self, wrapper):
        wrapper._method.execute.side_effect = RuntimeError("inference failed")
        with pytest.raises(RuntimeError, match="inference failed"):
            wrapper.call(torch.zeros(1, 3, 4, 4))
        wrapper._logger.warning.assert_called_once_with(
            "SuperPoint ExecuTorch inference error: inference failed"
        )
