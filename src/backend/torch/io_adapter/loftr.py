import cv2 as cv
import torch

from src.backend.io_adapter_base import IOAdapter
from src.backend.torch.io_adapter.io_adapter import TorchIOAdapter
from src.backend.torch.utils import image_to_tensor, matches_to_cv


@IOAdapter.register("loftr_torch")
class LoFTRTorchIOAdapter(TorchIOAdapter):
    def _preprocess_image(self, img):
        if torch.is_tensor(img):
            if img.ndim == 4:
                img = img.squeeze(0)

            if img.shape[0] == 3:
                img_np = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
                img_np = img_np.cpu().numpy()
            else:
                img_np = img.squeeze(0).cpu().numpy()
        else:
            if img.ndim == 3:
                img_np = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            else:
                img_np = img

        h, w = img_np.shape[:2]
        max_dim = 640
        scale = max_dim / max(h, w)
        if scale < 1.0:
            h, w = int(h * scale), int(w * scale)

        new_h, new_w = (h // 8) * 8, (w // 8) * 8

        if img_np.shape[0] != new_h or img_np.shape[1] != new_w:
            img_np = cv.resize(img_np, (new_w, new_h), interpolation=cv.INTER_AREA)

        tensor = torch.from_numpy(img_np).float()
        if tensor.max() > 1.1:
            tensor /= 255.0

        tensor = tensor.unsqueeze(0).unsqueeze(0)
        return tensor.to(self._device)

    def preprocess(self, inputs):
        img0 = inputs.pop('image0')
        img1 = inputs.pop('image1')

        img0 = image_to_tensor(img0)
        img1 = image_to_tensor(img1)

        img0 = self._preprocess_image(img0)
        img1 = self._preprocess_image(img1)
        return {'image0': img0, 'image1': img1}

    def postprocess(self, outputs):
        keypoints0 = outputs['mkpts0_f'].detach().cpu()
        keypoints1 = outputs['mkpts1_f'].detach().cpu()
        scores = outputs['mconf'].detach().cpu()

        num_matches = len(keypoints0)
        indices = torch.arange(num_matches, dtype=torch.long).view(-1, 1).repeat(1, 2)

        data = {"keypoints0": keypoints0, "keypoints1": keypoints1,
                "matches" : indices, "scores" : scores}
        return matches_to_cv(data)
