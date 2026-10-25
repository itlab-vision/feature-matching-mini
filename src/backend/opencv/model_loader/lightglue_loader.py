import cv2 as cv
from src.backend.model_wrapper_base import ModelWrapper
from src.backend.opencv.model_loader.model_loader import OpenCVModelLoader


@ModelWrapper.register("lightglue_opencv")
class LightGlueOpenCVModelLoader(OpenCVModelLoader):
    def load(self):
        checkpoint = self._config.pop('lightglue_model_path', "models/disk_lightglue_2outputs.onnx")
        self._logger.info(f"Initializing LightGlue from {checkpoint}")
        self._model = cv.dnn.readNet(checkpoint)
        return self._model
