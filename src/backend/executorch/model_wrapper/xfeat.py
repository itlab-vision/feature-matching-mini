import torch

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.executorch.model_wrapper.model_wrapper import ExecuTorchModelWrapper


@ModelWrapper.register("xfeat_executorch")
class XFeatExecuTorchModelWrapper(ExecuTorchModelWrapper):
    def load(self):
        return super().load()

    def call(self, inputs):
        if self._method is None:
            raise RuntimeError("ExecuTorch model is not loaded. Call load() first.")

        if isinstance(inputs, dict):
            image = inputs.get("image")
        else:
            image = inputs

        if not isinstance(image, torch.Tensor):
            raise TypeError("XFeat ExecuTorch expects a torch.Tensor input")

        try:
            with torch.no_grad():
                return self._method.execute((image.contiguous().cpu(),))
        except Exception as error:
            self._logger.warning(f"XFeat ExecuTorch inference error: {error}")
            raise
