from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.backend.executorch.model_wrapper.lightglue_matcher import (
    LightGlueExecuTorchModelWrapper,
)
from src.backend.model_wrapper_base import ModelWrapper


@pytest.fixture
def wrapper():
    wrapper = LightGlueExecuTorchModelWrapper("lightglue_executorch", config={"num_keypoints": 2}, logger=MagicMock())
    wrapper._method = MagicMock()
    return wrapper


@pytest.fixture
def inputs():
    return {
        f"features{i}": {
            "keypoints": np.full((2, 2), i, dtype=np.float32),
            "descriptors": np.full((2, 4), i + 2, dtype=np.float32),
        }
        for i in range(2)
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
        expected = (torch.tensor([[0, -1]]), torch.tensor([[0, -1]]), torch.ones(1, 2), torch.ones(1, 2))
        wrapper._method.execute.return_value = expected

        result = wrapper.call(inputs)

        tensors = wrapper._method.execute.call_args.args[0]
        expected_tensors = [inputs[f"features{i}"][key][None]
                            for key in ("keypoints", "descriptors") for i in range(2)]
        assert len(tensors) == len(expected_tensors)
        for tensor, value in zip(tensors, expected_tensors):
            np.testing.assert_array_equal(tensor.numpy(), value)
        assert all(tensor.device.type == "cpu" and tensor.is_contiguous() for tensor in tensors)
        np.testing.assert_array_equal(result["matches"], [[0, 0]])
        np.testing.assert_array_equal(result["scores"], [1.0])

    def test_call_requires_loaded_model(self, inputs):
        wrapper = LightGlueExecuTorchModelWrapper("lightglue_executorch")
        with pytest.raises(RuntimeError, match="load"):
            wrapper.call(inputs)

    def test_call_requires_features(self, wrapper):
        with pytest.raises(KeyError, match="features0"):
            wrapper.call({})

    def test_call_checks_descriptor_count(self, wrapper, inputs):
        inputs["features1"]["descriptors"] = np.zeros((1, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="expects 2 descriptors"):
            wrapper.call(inputs)

    def test_call_logs_and_reraises_runtime_error(self, wrapper, inputs):
        wrapper._method.execute.side_effect = RuntimeError("inference failed")
        with pytest.raises(RuntimeError, match="inference failed"):
            wrapper.call(inputs)
        wrapper._logger.warning.assert_called_once_with(
            "LightGlue ExecuTorch inference error: inference failed"
        )
