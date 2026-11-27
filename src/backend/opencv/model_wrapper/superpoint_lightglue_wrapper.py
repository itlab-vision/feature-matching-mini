import cv2 as cv

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.opencv.model_wrapper.model_wrapper import OpenCVModelWrapper


@ModelWrapper.register("superpoint_lightglue_opencv")
class SuperPointLightGlueOpenCVModelWrapper(OpenCVModelWrapper):
    def load(self):
        model_path = self._config.get("superpoint_model_path", "models/superpoint_lightglue.onnx")
        self._logger.info(f"Initializing SuperPoint from {model_path}")
        self._model = cv.dnn.readNet(str(model_path))
        return {"model": self._model}
