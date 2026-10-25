import cv2 as cv
import torch
import numpy as np
from PIL import Image

from src.backend.io_adapter_base import IOAdapter
from src.backend.torch.io_adapter.io_adapter import TorchIOAdapter
from src.backend.torch.utils import image_to_tensor, matches_to_cv


@IOAdapter.register("roma_torch")
class RoMaTorchIOAdapter(TorchIOAdapter):
    def _preprocess_image(self, img):
        if torch.is_tensor(img):
            if img.ndim == 4:
                img = img.squeeze(0)
            img_np = img.permute(1, 2, 0).cpu().detach().numpy()

            if img_np.max() <= 1.01:
                img_np = (img_np * 255).astype(np.uint8)
            return Image.fromarray(img_np)
        img_np = np.array(img)

        if len(img_np.shape) == 3:
            img_rgb = cv.cvtColor(img_np.astype(np.uint8), cv.COLOR_BGR2RGB)
        else:
            img_rgb = cv.cvtColor(img_np.astype(np.uint8), cv.COLOR_GRAY2RGB)
        return Image.fromarray(img_rgb)

    def preprocess(self, inputs):
        img0 = inputs.pop('image0')
        img1 = inputs.pop('image1')

        img0 = image_to_tensor(img0)
        img1 = image_to_tensor(img1)

        pil0 = self._preprocess_image(img0)
        pil1 = self._preprocess_image(img1)

        w0_orig, h0_orig = pil0.size
        w1_orig, h1_orig = pil1.size
        return {'image0': pil0, 'image1': pil1, 'width0': w0_orig, 'width1': w1_orig,
                'height0': h0_orig, 'height1': h1_orig, **inputs}

    def postprocess(self, outputs):
        keypoints0 = outputs.get('keypoints0').cpu()
        keypoints1 = outputs.get('keypoints1').cpu()
        scores = outputs.get('scores').cpu()

        num_matches = len(keypoints0)
        indices = torch.arange(num_matches, dtype=torch.long).view(-1, 1).repeat(1, 2)

        data = {"keypoints0": keypoints0, "keypoints1": keypoints1,
                "matches": indices, "scores": scores}
        return matches_to_cv(data)
