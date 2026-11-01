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
    def test_call_executes_image_tensor(self, wrapper):
        image = torch.zeros(1, 3, 4, 4).transpose(2, 3)
        expected = (torch.ones(1, 4, 2, 2),)
        wrapper._method.execute.return_value = expected

        result = wrapper.call({"image": image})

        executed = wrapper._method.execute.call_args.args[0][0]
        assert executed.device.type == "cpu"
        assert executed.is_contiguous()
        assert result is expected

    def test_call_accepts_direct_tensor(self, wrapper):
        image = torch.zeros(1, 3, 4, 4)
        wrapper.call(image)
        wrapper._method.execute.assert_called_once()

    def test_call_requires_loaded_model(self):
        with pytest.raises(RuntimeError, match="load"):
            D2NetExecuTorchModelWrapper("d2net_executorch").call(torch.zeros(1))

    def test_call_rejects_non_tensor(self, wrapper):
        with pytest.raises(TypeError, match="torch.Tensor"):
            wrapper.call({"image": "image"})

    def test_call_logs_and_reraises_runtime_error(self, wrapper):
        wrapper._method.execute.side_effect = RuntimeError("inference failed")
        with pytest.raises(RuntimeError, match="inference failed"):
            wrapper.call(torch.zeros(1, 3, 4, 4))
        wrapper._logger.warning.assert_called_once_with(
            "D2-Net ExecuTorch inference error: inference failed"
        )
