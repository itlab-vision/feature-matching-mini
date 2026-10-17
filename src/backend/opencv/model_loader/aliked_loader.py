import cv2 as cv
from src.backend.model_wrapper_base import ModelWrapper
from src.backend.opencv.model_loader.model_loader import OpenCVModelLoader


@ModelWrapper.register("aliked_opencv")
class AlikedOpenCVModelLoader(OpenCVModelLoader):
    def load(self):
        checkpoint = self._config.pop('aliked_model_path', "models/aliked-n32-top2k-640.onnx")
        self._logger.info(f"Initializing Aliked from {checkpoint}")
        self._model = cv.dnn.readNet(checkpoint)
        return self._model
