from pathlib import Path

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F

from thirdparty.tfeat.tfeat_model import TNet
from src.backend.model_wrapper_base import ModelWrapper
from src.backend.torch.model_wrapper.model_wrapper import TorchModelWrapper


@ModelWrapper.register('tfeat_torch')
class TFeatTorchModelWrapper(TorchModelWrapper):
    def load(self):
        path = Path(self._model_path or self._config.get(
            'tfeat_model_path', 'thirdparty/tfeat/pretrained-models/tfeat-liberty.params'))
        self._model = TNet().to(self._device)
        self._model.load_state_dict(torch.load(path, map_location=self._device, weights_only=True))
        self._model.eval()
        return {'model': self._model}

    def preprocess(self, inputs):
        image = inputs['image']
        if image.ndim == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = torch.from_numpy(np.ascontiguousarray(image)).float()[None, None].to(self._device) / 255.0
        keypoints = inputs['keypoints']
        size = self._config.get('patch_size', 32)
        if len(keypoints) == 0:
            return torch.empty((0, 1, size, size), device=self._device), keypoints, inputs['scores']
        xy = torch.tensor([point.pt for point in keypoints], dtype=torch.float32, device=self._device)
        offsets = (torch.arange(size, dtype=torch.float32, device=self._device) - size / 2 + 0.5)
        offsets = offsets * self._config.get('magfactor', 1)
        grid_y, grid_x = torch.meshgrid(offsets, offsets, indexing='ij')
        grid = xy[:, None, None] + torch.stack((grid_x, grid_y), dim=-1)
        height, width = image.shape[-2:]
        grid[..., 0] = grid[..., 0] / max(width - 1, 1) * 2 - 1
        grid[..., 1] = grid[..., 1] / max(height - 1, 1) * 2 - 1
        patches = F.grid_sample(image.expand(len(xy), -1, -1, -1), grid, align_corners=True)
        return patches.contiguous(), keypoints, inputs['scores']

    @staticmethod
    def postprocess(descriptors, keypoints, scores):
        return {'keypoints': keypoints, 'descriptors': descriptors.detach().cpu().numpy(), 'scores': scores}

    def call(self, inputs):
        patches, keypoints, scores = self.preprocess(inputs)
        with torch.no_grad():
            descriptors = self._model(patches) if len(patches) else torch.empty((0, 128))
        return self.postprocess(descriptors, keypoints, scores)
