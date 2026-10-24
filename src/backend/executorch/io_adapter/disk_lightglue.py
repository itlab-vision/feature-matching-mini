from src.backend.io_adapter_base import IOAdapter
from src.backend.executorch.io_adapter.io_adapter import ExecuTorchIOAdapter
from src.backend.executorch.utils import image_to_tensor


@IOAdapter.register("disk_lightglue_executorch")
class DiskLightGlueExecuTorchIOAdapter(ExecuTorchIOAdapter):
    def preprocess(self, inputs):
        img = inputs.pop('image')
        img = image_to_tensor(img)

        return {'image': img, **inputs}

    def postprocess(self, outputs):
        keypoints = outputs.get('keypoints')
        descriptors = outputs.get('descriptors')
        scores = outputs.get('scores')

        result = {'keypoints': keypoints, 'descriptors': descriptors, 'scores': scores}
        return result
