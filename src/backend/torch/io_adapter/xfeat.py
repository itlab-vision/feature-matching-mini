import cv2 as cv
import torch

from src.backend.io_adapter_base import IOAdapter
from src.backend.torch.io_adapter.io_adapter import TorchIOAdapter
from src.backend.torch.utils import to_numpy_bgr, features_to_cv


@IOAdapter.register("xfeat_torch")
class XFeatTorchIOAdapter(TorchIOAdapter):
    def preprocess(self, inputs):
        img = inputs.pop("image")
        input_type = 'torch' if isinstance(img, torch.Tensor) else 'numpy'
        img_np = to_numpy_bgr(img, input_type=input_type)
        img_rgb = cv.cvtColor(img_np, cv.COLOR_BGR2RGB)

        input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0)
        img = input_tensor.to(self._device) / 255.0
        return {'image': img, **inputs}

    def postprocess(self, outputs):
        return features_to_cv(outputs)
