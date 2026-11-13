import cv2 as cv

from src.backend.io_adapter_base import IOAdapter
from src.backend.opencv.io_adapter.io_adapter import OpenCVIOAdapter
from src.backend.superpoint_dense import postprocess_superpoint_dense


@IOAdapter.register("superpoint_lightglue_opencv")
class SuperPointLightGlueOpenCVIOAdapter(OpenCVIOAdapter):
    def __init__(self, model_name, model_wrapper=None, logger=None, config=None):
        super().__init__(model_name, model_wrapper, logger, config)
        self._input_shape = self._config.get("input_shape", (1, 3, 480, 640))
        self._num_keypoints = self._config.get("num_keypoints", 512)
        self._nms_radius = self._config.get("nms_radius", 4)

    def preprocess(self, inputs):
        image = inputs.pop("image")
        self._original_size = image.shape[:2]
        height, width = self._input_shape[-2:]
        blob = cv.dnn.blobFromImage(image, scalefactor=1.0 / 255.0, size=(width, height), swapRB=True, crop=False)
        self._model.setInput(blob)
        return {"image": blob}

    def postprocess(self, outputs):
        return postprocess_superpoint_dense(
            outputs, self._original_size, self._input_shape[-2:], self._num_keypoints, self._nms_radius
        )
