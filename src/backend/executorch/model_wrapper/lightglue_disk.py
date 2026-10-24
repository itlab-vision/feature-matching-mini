from src.backend.model_wrapper_base import ModelWrapper
from src.backend.executorch.model_wrapper.model_wrapper import ExecuTorchModelWrapper


@ModelWrapper.register("lightglue_diks_executorch")
class LightGlueDiskExecuTorchModelWrapper(ExecuTorchModelWrapper):
    def load(self):
        pass

    def call(self, inputs):
        pass
