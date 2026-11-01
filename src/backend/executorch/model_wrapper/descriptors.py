import torch
from src.backend.executorch.model_wrapper.model_wrapper import ExecuTorchModelWrapper

from src.backend.model_wrapper_base import ModelWrapper


@ModelWrapper.register("tfeat_executorch")
@ModelWrapper.register("hardnet_executorch")
class PatchDescriptorExecuTorchModelWrapper(ExecuTorchModelWrapper):
    def call(self, inputs):
        if self._method is None:
            raise RuntimeError("ExecuTorch model is not loaded. Call load() first.")

        patches = inputs.get("patches") if isinstance(inputs, dict) else inputs
        if not isinstance(patches, torch.Tensor) or patches.ndim != 4:
            raise TypeError("Patch descriptor ExecuTorch model expects a [N, C, H, W] torch.Tensor")

        if patches.shape[0] == 0:
            descriptor_dim = self._config.get("descriptor_dim", 128)
            return torch.empty((0, descriptor_dim), dtype=torch.float32)

        descriptors = []
        try:
            with torch.no_grad():
                for index in range(patches.shape[0]):
                    patch = patches[index:index + 1].contiguous().cpu()
                    output = self._method.execute((patch,))
                    descriptor = output[0] if isinstance(output, (tuple, list)) else output
                    descriptors.append(descriptor)
        except Exception as error:
            if self._logger:
                self._logger.warning(f"{self._model_name} ExecuTorch inference error: {error}")
            raise

        return torch.cat(descriptors, dim=0)
