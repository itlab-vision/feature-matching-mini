from unittest.mock import MagicMock, patch

import pytest
import torch

from src.backend.executorch.model_wrapper.model_wrapper import ExecuTorchModelWrapper
from src.backend.model_wrapper_base import ModelWrapper


@pytest.fixture
def wrapper(tmp_path):
    model_path = tmp_path / "model.pte"
    model_path.touch()
    return ExecuTorchModelWrapper(
        "executorch",
        model_path=model_path,
        config={},
        logger=MagicMock(),
    )


class TestExecuTorchModelWrapperRegistration:
    def test_registered_in_factory(self):
        assert ModelWrapper._METHODS["executorch"] is ExecuTorchModelWrapper


class TestExecuTorchModelWrapperInit:
    def test_accepts_none_config(self):
        wrapper = ExecuTorchModelWrapper("executorch")
        assert wrapper._config == {}
        assert wrapper._program is None
        assert wrapper._method is None


class TestExecuTorchModelWrapperLoad:
    @patch("src.backend.executorch.model_wrapper.model_wrapper.Runtime")
    def test_loads_program_and_forward_method(self, mock_runtime, wrapper):
        method = MagicMock()
        program = mock_runtime.get.return_value.load_program.return_value
        program.load_method.return_value = method

        result = wrapper.load()

        mock_runtime.get.return_value.load_program.assert_called_once()
        assert mock_runtime.get.return_value.load_program.call_args.args[0] == str(wrapper._model_path)
        program.load_method.assert_called_once_with("forward")
        assert result == {"model": method}
        assert wrapper.model is method
        wrapper._logger.info.assert_called_once()

    @patch("src.backend.executorch.model_wrapper.model_wrapper.Runtime")
    def test_uses_configured_model_path_and_method(self, mock_runtime, tmp_path):
        model_path = tmp_path / "configured.pte"
        model_path.touch()
        wrapper = ExecuTorchModelWrapper(
            "executorch",
            config={"executorch_model_path": model_path, "method_name": "encode"},
        )

        wrapper.load()

        mock_runtime.get.return_value.load_program.assert_called_once()
        program = mock_runtime.get.return_value.load_program.return_value
        program.load_method.assert_called_once_with("encode")

    def test_requires_model_path(self):
        with pytest.raises(ValueError, match="model_path"):
            ExecuTorchModelWrapper("executorch").load()

    def test_requires_existing_model(self, tmp_path):
        wrapper = ExecuTorchModelWrapper("executorch", model_path=tmp_path / "missing.pte")
        with pytest.raises(FileNotFoundError, match="does not exist"):
            wrapper.load()

    @patch("src.backend.executorch.model_wrapper.model_wrapper.Runtime", None)
    def test_requires_executorch_runtime_for_existing_model(self, wrapper):
        with pytest.raises(ModuleNotFoundError, match="ExecuTorch runtime is not installed"):
            wrapper.load()


class TestExecuTorchModelWrapperCall:
    def test_call_wraps_single_input_in_tuple(self, wrapper):
        wrapper._method = MagicMock()
        output = (torch.tensor([1.0]),)
        wrapper._method.execute.return_value = output
        image = torch.zeros(1, 3, 4, 4)

        result = wrapper.call(image)

        wrapper._method.execute.assert_called_once_with((image,))
        assert result is output

    def test_call_preserves_input_sequence(self, wrapper):
        wrapper._method = MagicMock()
        inputs = [torch.tensor(1), torch.tensor(2)]
        wrapper.call(inputs)
        wrapper._method.execute.assert_called_once_with(tuple(inputs))

    def test_call_requires_loaded_model(self, wrapper):
        with pytest.raises(RuntimeError, match="load"):
            wrapper.call(torch.zeros(1))
