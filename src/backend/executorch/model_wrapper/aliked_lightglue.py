from src.backend.model_wrapper_base import ModelWrapper
from src.backend.executorch.model_wrapper.disk_lightglue import DiskLightGlueExecuTorchModelWrapper


@ModelWrapper.register("aliked_lightglue_executorch")
class AlikedLightGlueExecuTorchModelWrapper(DiskLightGlueExecuTorchModelWrapper):
    pass
