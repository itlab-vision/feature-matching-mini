import torch

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.executorch.model_wrapper.model_wrapper import ExecuTorchModelWrapper


@ModelWrapper.register("d2net_executorch")
class D2NetExecuTorchModelWrapper(ExecuTorchModelWrapper):
    def load(self):
        return super().load()

    def call(self, inputs):
        if self._method is None:
            raise RuntimeError("ExecuTorch model is not loaded. Call load() first.")

        image = inputs.get("image") if isinstance(inputs, dict) else inputs
        if not isinstance(image, torch.Tensor):
            raise TypeError("D2-Net ExecuTorch expects a torch.Tensor input")

        try:
            with torch.no_grad():
                return self._method.execute((image.contiguous().cpu(),))
        except Exception as error:
            if self._logger:
                self._logger.warning(f"D2-Net ExecuTorch inference error: {error}")
            raise
