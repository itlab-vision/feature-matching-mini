import torch

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.executorch.model_wrapper.model_wrapper import ExecuTorchModelWrapper


@ModelWrapper.register("lightglue_disk_executorch")
@ModelWrapper.register("lightglue_superpoint_executorch")
@ModelWrapper.register("lightglue_executorch")
class LightGlueExecuTorchModelWrapper(ExecuTorchModelWrapper):
    def load(self):
        return super().load()

    def call(self, inputs):
        if self._method is None:
            raise RuntimeError("ExecuTorch model is not loaded. Call load() first.")
        if not isinstance(inputs, dict):
            raise TypeError("LightGlue ExecuTorch expects a dictionary of feature tensors")

        names = ["keypoints0", "keypoints1", "descriptors0", "descriptors1"]
        descriptor_class = self._config.get("descriptor_class")
        descriptor_name = descriptor_class._descriptor_name if descriptor_class is not None else ""
        if "sift" in descriptor_name or "doghardnet" in descriptor_name:
            names.extend(("scales0", "oris0", "scales1", "oris1"))
        tensors = tuple(inputs.get(name) for name in names)
        if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError(f"LightGlue ExecuTorch expects torch tensors: {names}")

        tensors = tuple(tensor.contiguous().cpu() for tensor in tensors)
        try:
            with torch.no_grad():
                return self._method.execute(tensors)
        except Exception as error:
            if self._logger:
                self._logger.warning(f"LightGlue ExecuTorch inference error: {error}")
            raise
