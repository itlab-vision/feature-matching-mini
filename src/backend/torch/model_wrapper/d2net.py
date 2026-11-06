from pathlib import Path
import torch
import numpy as np
import cv2 as cv

from thirdparty.d2net.lib.utils import preprocess_image
from thirdparty.d2net.lib.model_test import D2Net as D2NetModel
from thirdparty.d2net.lib.pyramid import process_multiscale

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.torch.model_wrapper.model_wrapper import TorchModelWrapper
from src.backend.torch.model_wrapper.weights_url import WEIGHTS_URL_MODELS


@ModelWrapper.register("d2net_torch")
class D2NetTorchModelWrapper(TorchModelWrapper):
    def load(self):
        checkpoint = Path(self._model_path or self._config.pop('checkpoint', "weights/d2net/d2_tf.pth"))
        weights_url = WEIGHTS_URL_MODELS.get(self._model_name)

        use_relu = self._config.get('use_relu', True)
        device_str = self._config.get('device', 'cpu').lower()
        use_cuda = 'cuda' in device_str

        if not checkpoint.exists():
            self._logger.info(f"Downloading D2Net weights to {checkpoint}")
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(weights_url, str(checkpoint))

        self._logger.info(f"Initializing D2NetModel from {checkpoint}")
        self._model = D2NetModel(model_file=str(checkpoint), use_relu=use_relu, use_cuda=use_cuda)
        self._model = self._model.to(self._device)
        self._model.eval()
        return {'model': self._model}

    def preprocess(self, inputs):
        image = cv.cvtColor(np.asarray(inputs["image"]), cv.COLOR_BGR2RGB)
        image = preprocess_image(image, preprocessing='caffe')
        return torch.from_numpy(image).float().unsqueeze(0).to(self._device)

    @staticmethod
    def postprocess(outputs):
        keypoints, scores, descriptors = outputs
        return {'keypoints': keypoints[:, [1, 0]].astype(np.float32, copy=False),
                'descriptors': descriptors,
                'scores': scores}

    def call(self, inputs):
        try:
            image = self.preprocess(inputs)
            with torch.no_grad():
                outputs = process_multiscale(image, self._model, scales=[1])
            return self.postprocess(outputs)
        except Exception as error:
            self._logger.error(f"D2-Net inference error: {error}")
            raise
