from src.backend.io_adapter_base import IOAdapter
from src.backend.executorch.io_adapter.io_adapter import ExecuTorchIOAdapter
from src.backend.executorch.utils import image_to_tensor


@IOAdapter.register("lightglue_disk_executorch")
class LightGlueDiskExecuTorchIOAdapter(ExecuTorchIOAdapter):
    def preprocess(self, inputs):
        pass

    def postprocess(self, outputs):
        pass