import cv2 as cv
from src.backend.model_wrapper_base import ModelWrapper
from src.backend.opencv.model_wrapper.model_wrapper import OpenCVModelWrapper


@ModelWrapper.register("lightglue_opencv")
class LightGlueOpenCVModelWrapper(OpenCVModelWrapper):
    def load(self):
        descriptor = self._config.get('descriptor_class')
        feature = descriptor._descriptor_name.removesuffix('_lightglue') if descriptor is not None else 'disk'
        checkpoint = self._config.pop('lightglue_model_path', None) or f'models/lightglue_{feature}_opencv.onnx'
        self._logger.info(f"Initializing LightGlue from {checkpoint}")
        self._model = cv.dnn.readNet(str(checkpoint))
        return {'model': self._model}
