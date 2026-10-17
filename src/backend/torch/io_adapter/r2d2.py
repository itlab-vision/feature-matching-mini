import cv2 as cv
import torch
import numpy as np

from src.backend.io_adapter_base import IOAdapter
from src.backend.torch.io_adapter.io_adapter import TorchIOAdapter
from src.backend.torch.utils import image_to_tensor, features_to_cv


@IOAdapter.register("r2d2_torch")
class R2D2TorchIOAdapter(TorchIOAdapter):
    def preprocess(self, inputs):
        img = inputs.pop('image')
        img = image_to_tensor(img)

        if torch.is_tensor(img):
            img_np = img.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
            if img_np.max() <= 1.01:
                img_np *= 255.0
        else:
            img_np = np.array(img)

        img_rgb = cv.cvtColor(img_np.astype(np.uint8), cv.COLOR_BGR2RGB)

        input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        input_tensor = (input_tensor - mean) / std
        input_tensor = input_tensor.unsqueeze(0).to(self._device)
        return {'image': input_tensor, **inputs}

    def postprocess(self, outputs):
        xys = outputs.get('keypoints')
        xys = xys[:, :2]
        outputs['keypoints'] = xys
        return features_to_cv(outputs)
