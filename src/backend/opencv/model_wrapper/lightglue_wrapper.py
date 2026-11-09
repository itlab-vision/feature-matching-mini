import cv2 as cv
from src.backend.model_wrapper_base import ModelWrapper
from src.backend.model_paths import ONNX_MATCHER_MODELS
from src.backend.opencv.model_wrapper.model_wrapper import OpenCVModelWrapper


@ModelWrapper.register("lightglue_opencv")
class LightGlueOpenCVModelWrapper(OpenCVModelWrapper):
    def load(self):
        checkpoint = self._config.pop('lightglue_model_path', f"models/{ONNX_MATCHER_MODELS['disk']}")
        self._logger.info(f"Initializing LightGlue from {checkpoint}")
        self._model = cv.dnn.readNet(checkpoint)
        return {'model': self._model}
