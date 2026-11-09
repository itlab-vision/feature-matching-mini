import cv2 as cv
from src.backend.model_wrapper_base import ModelWrapper
from src.backend.model_paths import ONNX_DETECTOR_MODELS
from src.backend.opencv.model_wrapper.model_wrapper import OpenCVModelWrapper


@ModelWrapper.register("aliked_opencv")
class AlikedOpenCVModelWrapper(OpenCVModelWrapper):
    def load(self):
        checkpoint = self._config.pop('aliked_model_path', f"models/{ONNX_DETECTOR_MODELS['aliked']}")
        self._logger.info(f"Initializing Aliked from {checkpoint}")
        self._model = cv.dnn.readNet(checkpoint)
        return {'model': self._model}
