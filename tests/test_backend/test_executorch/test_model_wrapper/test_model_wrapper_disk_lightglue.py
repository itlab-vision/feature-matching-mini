from unittest.mock import MagicMock, patch

import numpy as np
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
    wrapper._config["input_shape"] = (1, 3, 8, 8)
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
    def test_call_executes_contiguous_cpu_image(self, wrapper):
        image = np.zeros((4, 4, 3), dtype=np.uint8)[:, ::-1]
        expected = (torch.zeros(1, 1, 4, 4), torch.zeros(1, 8, 4, 4))
        wrapper._method.execute.return_value = expected

        result = wrapper.call({"image": image})

        executed = wrapper._method.execute.call_args.args[0][0]
        assert executed.device.type == "cpu"
        assert executed.is_contiguous()
        assert set(result) == {"keypoints", "descriptors", "scores"}
        assert all(isinstance(value, np.ndarray) for value in result.values())
        assert result["keypoints"].shape == (len(result["scores"]), 2)
        assert len(result["descriptors"]) == len(result["scores"])

    def test_call_requires_loaded_model(self):
        with pytest.raises(RuntimeError, match="load"):
            DiskLightGlueExecuTorchModelWrapper("disk_lightglue_executorch").call(torch.zeros(1))

    def test_call_requires_image(self, wrapper):
        with pytest.raises(KeyError, match="image"):
            wrapper.call({})

    def test_call_logs_and_reraises_runtime_error(self, wrapper):
        wrapper._method.execute.side_effect = RuntimeError("inference failed")
        with pytest.raises(RuntimeError, match="inference failed"):
            wrapper.call({"image": np.zeros((4, 4, 3), dtype=np.uint8)})
        wrapper._logger.warning.assert_called_once_with(
            "DISK ExecuTorch inference error: inference failed"
        )
