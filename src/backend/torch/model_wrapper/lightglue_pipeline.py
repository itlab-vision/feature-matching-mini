import torch
import cv2 as cv
import numpy as np

from thirdparty.lightglue.lightglue import SuperPoint, DISK, SIFT, ALIKED, DoGHardNet

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.torch.model_wrapper.model_wrapper import TorchModelWrapper


@ModelWrapper.register("light_glue_feature_extractor_torch")
class LightGlueFeatureExtractorTorchModelWrapper(TorchModelWrapper):
    _EXTRACTOR_CLASSES = {
        'superpoint_lightglue_torch': SuperPoint,
        'disk_lightglue_torch': DISK,
        'sift_lightglue_torch': SIFT,
        'aliked_lightglue_torch': ALIKED,
        'doghardnet_lightglue_torch': DoGHardNet
    }
    _shared_models = {}

    def load(self):
        model_key = (self._model_name, self._device.type)
        if model_key not in LightGlueFeatureExtractorTorchModelWrapper._shared_models:
            extractor_class = self._EXTRACTOR_CLASSES.get(f"{self._model_name}_torch")
            if not extractor_class:
                raise ValueError(f"Extractor '{self._model_name}' not found.")

            self._logger.info(f"Loading {self._model_name} weights onto {self._device}")
            LightGlueFeatureExtractorTorchModelWrapper._shared_models[model_key] = \
                (extractor_class(**self._config).eval().to(self._device))

        self._model = LightGlueFeatureExtractorTorchModelWrapper._shared_models[model_key]
        return {'model': self._model}

    def preprocess(self, inputs):
        rgb = cv.cvtColor(inputs['image'], cv.COLOR_BGR2RGB)
        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
        return image.unsqueeze(0).to(self._device)

    @staticmethod
    def postprocess(outputs):
        result = {key: value.detach().cpu().numpy() if torch.is_tensor(value) else value
                  for key, value in outputs.items()}
        result['keypoints'] = outputs['keypoints'].squeeze(0).detach().cpu().numpy()
        result['descriptors'] = outputs['descriptors'].squeeze(0).detach().cpu().numpy()
        result['scores'] = outputs['keypoint_scores'].squeeze(0).detach().cpu().numpy()
        result.pop('keypoint_scores', None)
        return result

    def call(self, inputs):
        try:
            image = self.preprocess(inputs)
            with torch.no_grad():
                self._logger.info(f"Running inference with {self._model_name} on torch backend")
                outputs = self._model.extract(image)
            return self.postprocess(outputs)
        except Exception as error:
            self._logger.error(f"LightGlue extractor inference error: {error}")
            raise
