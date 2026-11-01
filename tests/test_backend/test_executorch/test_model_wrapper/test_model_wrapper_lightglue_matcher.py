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
    def test_call_executes_feature_tensors_in_model_order(self, wrapper, inputs):
        expected = tuple(torch.zeros(1, 2) for _ in range(4))
        wrapper._method.execute.return_value = expected

        result = wrapper.call(inputs)

        tensors = wrapper._method.execute.call_args.args[0]
        names = ("keypoints0", "keypoints1", "descriptors0", "descriptors1")
        assert all(tensor is inputs[name] for tensor, name in zip(tensors, names))
        assert all(tensor.device.type == "cpu" and tensor.is_contiguous() for tensor in tensors)
        assert result is expected

    def test_call_requires_loaded_model(self, inputs):
        wrapper = LightGlueExecuTorchModelWrapper("lightglue_executorch")
        with pytest.raises(RuntimeError, match="load"):
            wrapper.call(inputs)

    def test_call_requires_dictionary(self, wrapper):
        with pytest.raises(TypeError, match="dictionary"):
            wrapper.call([])

    def test_call_requires_all_tensor_inputs(self, wrapper, inputs):
        inputs["descriptors1"] = None
        with pytest.raises(TypeError, match="torch tensors"):
            wrapper.call(inputs)

    def test_call_logs_and_reraises_runtime_error(self, wrapper, inputs):
        wrapper._method.execute.side_effect = RuntimeError("inference failed")
        with pytest.raises(RuntimeError, match="inference failed"):
            wrapper.call(inputs)
        wrapper._logger.warning.assert_called_once_with(
            "LightGlue ExecuTorch inference error: inference failed"
        )
