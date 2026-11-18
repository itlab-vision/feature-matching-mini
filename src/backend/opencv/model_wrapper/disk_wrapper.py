import cv2 as cv
from src.backend.model_wrapper_base import ModelWrapper
from src.backend.model_paths import ONNX_DETECTOR_MODELS
from src.backend.opencv.model_wrapper.model_wrapper import OpenCVModelWrapper


@ModelWrapper.register("disk_opencv")
class DiskOpenCVModelWrapper(OpenCVModelWrapper):
    def load(self):
        checkpoint = self._config.pop('disk_model_path', f"models/{ONNX_DETECTOR_MODELS['disk']}")
        self._logger.info(f"Initializing Disk from {checkpoint}")
        self._model = cv.dnn.readNet(checkpoint)
        return {'model': self._model}
