import cv2 as cv
from src.backend.model_wrapper_base import ModelWrapper
from src.backend.opencv.model_loader.model_loader import (
    OPENCV_MATCHER_MODELS,
    OpenCVModelLoader,
)


@ModelWrapper.register("lightglue_opencv")
class LightGlueOpenCVModelLoader(OpenCVModelLoader):
    def load(self):
        checkpoint = self._config.pop('lightglue_model_path', f"models/{OPENCV_MATCHER_MODELS['disk']}")
        self._logger.info(f"Initializing LightGlue from {checkpoint}")
        self._model = cv.dnn.readNet(checkpoint)
        return {'model': self._model}
