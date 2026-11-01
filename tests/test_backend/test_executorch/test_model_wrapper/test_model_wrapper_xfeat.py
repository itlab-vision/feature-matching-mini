from unittest.mock import MagicMock, patch

import pytest
import torch

from src.backend.executorch.model_wrapper.xfeat import XFeatExecuTorchModelWrapper
from src.backend.model_wrapper_base import ModelWrapper


@pytest.fixture
def wrapper():
    wrapper = XFeatExecuTorchModelWrapper("xfeat_executorch", logger=MagicMock())
    wrapper._method = MagicMock()
    return wrapper


class TestXFeatExecuTorchModelWrapperRegistration:
    def test_registered_in_factory(self):
        assert ModelWrapper._METHODS["xfeat_executorch"] is XFeatExecuTorchModelWrapper


class TestXFeatExecuTorchModelWrapperLoad:
    @patch("src.backend.executorch.model_wrapper.xfeat.ExecuTorchModelWrapper.load")
    def test_load_delegates_to_base_wrapper(self, mock_load):
        expected = {"model": MagicMock()}
        mock_load.return_value = expected
        wrapper = XFeatExecuTorchModelWrapper("xfeat_executorch")
        assert wrapper.load() is expected


class TestXFeatExecuTorchModelWrapperCall:
    def test_call_executes_contiguous_cpu_image(self, wrapper):
        image = torch.zeros(1, 3, 4, 4).transpose(2, 3)
        expected = (torch.zeros(1, 65, 1, 1), torch.zeros(1, 4, 1, 1))
        wrapper._method.execute.return_value = expected

        result = wrapper.call({"image": image})

        executed = wrapper._method.execute.call_args.args[0][0]
        assert executed.device.type == "cpu"
        assert executed.is_contiguous()
        assert result is expected

    def test_call_accepts_direct_tensor(self, wrapper):
        wrapper.call(torch.zeros(1, 3, 4, 4))
        wrapper._method.execute.assert_called_once()

    def test_call_requires_loaded_model(self):
        with pytest.raises(RuntimeError, match="load"):
            XFeatExecuTorchModelWrapper("xfeat_executorch").call(torch.zeros(1))

    def test_call_rejects_non_tensor(self, wrapper):
        with pytest.raises(TypeError, match="torch.Tensor"):
            wrapper.call({"image": None})

    def test_call_logs_and_reraises_runtime_error(self, wrapper):
        wrapper._method.execute.side_effect = RuntimeError("inference failed")
        with pytest.raises(RuntimeError, match="inference failed"):
            wrapper.call(torch.zeros(1, 3, 4, 4))
        wrapper._logger.warning.assert_called_once_with(
            "XFeat ExecuTorch inference error: inference failed"
        )
