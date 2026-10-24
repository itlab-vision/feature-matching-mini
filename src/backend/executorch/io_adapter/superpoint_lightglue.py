from src.backend.io_adapter_base import IOAdapter
from src.backend.executorch.io_adapter.io_adapter import ExecuTorchIOAdapter
from src.backend.executorch.utils import image_to_tensor


@IOAdapter.register("superpoint_lightglue_executorch")
class SuperPointLightGlueExecuTorchIOAdapter(ExecuTorchIOAdapter):
    def preprocess(self, inputs):
        pass

    def postprocess(self, outputs):
        pass