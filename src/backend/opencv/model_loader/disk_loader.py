import cv2 as cv
from src.backend.model_wrapper_base import ModelWrapper
from src.backend.opencv.model_loader.model_loader import (
    OPENCV_DETECTOR_MODELS,
    OpenCVModelLoader,
)


@ModelWrapper.register("disk_opencv")
class DiskOpenCVModelLoader(OpenCVModelLoader):
    def load(self):
        checkpoint = self._config.pop('disk_model_path', f"models/{OPENCV_DETECTOR_MODELS['disk']}")
        self._logger.info(f"Initializing Disk from {checkpoint}")
        self._model = cv.dnn.readNet(checkpoint)
        return {'model': self._model}
