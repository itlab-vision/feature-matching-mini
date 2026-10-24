import cv2 as cv
import numpy as np
import torch

from src.backend.io_adapter_base import IOAdapter
from src.backend.executorch.utils import prepare_image_tensor


@IOAdapter.register("executorch")
class ExecuTorchIOAdapter(IOAdapter):
    def __init__(self, model_name, model_wrapper=None, logger=None, config=None):
        if config is None:
            config = {}

        super().__init__(model_name, model_wrapper, logger, config)
        self._input_shape = self._config.get("input_shape")

    def preprocess(self, inputs):
        if not isinstance(inputs, dict) or "image" not in inputs:
            raise ValueError("ExecuTorch adapter expects a dictionary containing 'image'")

        image = inputs["image"]
        if not isinstance(image, np.ndarray):
            raise TypeError("ExecuTorch adapter expects an OpenCV numpy image")

        if image.ndim == 2:
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        elif image.ndim != 3 or image.shape[2] != 1:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        if image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
        image = prepare_image_tensor(image, self._input_shape)

        return {**inputs, "image": image}

    def postprocess(self, outputs):
        if isinstance(outputs, dict):
            return outputs

        if isinstance(outputs, (tuple, list)):
            return tuple(outputs)

        return outputs
