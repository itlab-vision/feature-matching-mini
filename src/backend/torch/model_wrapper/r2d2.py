from pathlib import Path
import torch
import cv2 as cv
import numpy as np

from thirdparty.r2d2.extract import NonMaxSuppression, load_network
from thirdparty.r2d2.extract import extract_multiscale

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.torch.model_wrapper.model_wrapper import TorchModelWrapper


@ModelWrapper.register("r2d2_torch")
class R2D2TorchModelWrapper(TorchModelWrapper):
    def load(self):
        checkpoint = Path(self._model_path or self._config.pop('checkpoint', "thirdparty/r2d2/models/r2d2_WASF_N16.pt"))

        self._logger.info(f"Loading R2D2 weights from {checkpoint}")
        self._model = load_network(checkpoint).to(self._device).eval()
        self._nms = NonMaxSuppression()
        return {'model': self._model, 'nms': self._nms}

    def preprocess(self, inputs):
        image = cv.cvtColor(inputs['image'].astype(np.uint8), cv.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
        mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = tensor.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return ((tensor - mean) / std).unsqueeze(0).to(self._device)

    @staticmethod
    def postprocess(outputs):
        keypoints, descriptors, scores = outputs
        return {'keypoints': keypoints[:, :2].detach().cpu().numpy(),
                'descriptors': descriptors.detach().cpu().numpy(),
                'scores': scores.detach().cpu().numpy()}

    def call(self, inputs):
        try:
            image = self.preprocess(inputs)
            with torch.no_grad():
                outputs = extract_multiscale(self._model, image, self._nms)
            return self.postprocess(outputs)
        except Exception as error:
            self._logger.warning(f"R2D2 inference failed (likely 0 points): {error}")
            raise
