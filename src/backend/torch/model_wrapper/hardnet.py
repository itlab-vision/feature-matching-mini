from pathlib import Path

import cv2 as cv
import numpy as np
import torch

from thirdparty.hardnet.examples.extract_hardnet_desc_from_hpatches_file import HardNet as HardNetModel
from src.backend.model_wrapper_base import ModelWrapper
from src.backend.torch.model_wrapper.model_wrapper import TorchModelWrapper


@ModelWrapper.register('hardnet_torch')
class HardNetTorchModelWrapper(TorchModelWrapper):
    def load(self):
        path = Path(self._model_path or self._config.get(
            'hardnet_model_path', 'thirdparty/hardnet/pretrained/train_liberty_with_aug/'
                                  'checkpoint_liberty_with_aug.pth'))
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)
        weights = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self._model = HardNetModel().to(self._device)
        self._model.load_state_dict(weights)
        self._model.eval()
        return {'model': self._model}

    def preprocess(self, inputs):
        image = inputs['image']
        if image.ndim == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        height, width = image.shape
        size = self._config.get('patch_size', 32)
        half = size // 2
        patches, indices = [], []
        for index, point in enumerate(inputs['keypoints']):
            x, y = (int(round(value)) for value in point.pt)
            if x - half < 0 or x + half >= width or y - half < 0 or y + half >= height:
                continue
            patches.append(image[y - half:y + half, x - half:x + half])
            indices.append(index)
        indices = np.asarray(indices, dtype=np.int64)
        if not patches:
            tensor = torch.empty((0, 1, size, size), device=self._device)
        else:
            tensor = torch.from_numpy(np.stack(patches)).float().unsqueeze(1).to(self._device) / 255.0
            tensor = (tensor - 0.443728476019) / 0.20197947209
        return tensor, inputs['keypoints'][indices], inputs['scores'][indices]

    @staticmethod
    def postprocess(descriptors, keypoints, scores):
        return {'keypoints': keypoints, 'descriptors': descriptors.detach().cpu().numpy(), 'scores': scores}

    def call(self, inputs):
        patches, keypoints, scores = self.preprocess(inputs)
        descriptors = []
        with torch.no_grad():
            for start in range(0, len(patches), self._config.get('batch_size', 128)):
                descriptors.append(self._model(patches[start:start + self._config.get('batch_size', 128)]))
        output = torch.cat(descriptors) if descriptors else torch.empty((0, 128))
        return self.postprocess(output, keypoints, scores)
