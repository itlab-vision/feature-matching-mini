import cv2 as cv

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.opencv.model_loader.model_loader import OpenCVModelLoader


@ModelWrapper.register("superpoint_lightglue_opencv")
class SuperPointLightGlueOpenCVModelLoader(OpenCVModelLoader):
    def load(self):
        model_path = self._config.get("superpoint_model_path")
        if model_path is None:
            raise ValueError("SuperPoint OpenCV requires 'superpoint_model_path'")
        self._logger.info(f"Initializing SuperPoint from {model_path}")
        self._model = cv.dnn.readNet(model_path)
        return {"model": self._model}
