import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F

from src.backend.executorch.model_wrapper.model_wrapper import ExecuTorchModelWrapper
from src.backend.model_wrapper_base import ModelWrapper


@ModelWrapper.register('tfeat_executorch')
class TFeatExecuTorchModelWrapper(ExecuTorchModelWrapper):
    def preprocess(self, inputs):
        image = inputs['image']
        if image.ndim == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = torch.from_numpy(np.ascontiguousarray(image)).float()[None, None] / 255.0
        keypoints = inputs['keypoints']
        size = self._config.get('patch_size', 32)
        if len(keypoints) == 0:
            return torch.empty((0, 1, size, size)), keypoints, inputs['scores']
        xy = torch.tensor([point.pt for point in keypoints], dtype=torch.float32)
        offsets = (torch.arange(size, dtype=torch.float32) - size / 2 + 0.5)
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
        if self._method is None:
            raise RuntimeError("ExecuTorch model is not loaded. Call load() first.")
        patches, keypoints, scores = self.preprocess(inputs)
        descriptors = []
        try:
            with torch.no_grad():
                for index in range(patches.shape[0]):
                    patch = patches[index:index + 1].contiguous().cpu()
                    output = self._method.execute((patch,))
                    descriptor = output[0] if isinstance(output, (tuple, list)) else output
                    descriptors.append(descriptor)
        except Exception as error:
            if self._logger:
                self._logger.warning(f"{self._model_name} ExecuTorch inference error: {error}")
            raise

        output = torch.cat(descriptors) if descriptors else torch.empty(
            (0, self._config.get('descriptor_dim', 128)), dtype=torch.float32)
        return self.postprocess(output, keypoints, scores)
