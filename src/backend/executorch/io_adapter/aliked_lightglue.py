import cv2 as cv
import numpy as np
import torch

from src.backend.io_adapter_base import IOAdapter
from src.backend.executorch.io_adapter.io_adapter import ExecuTorchIOAdapter


@IOAdapter.register("aliked_lightglue_executorch")
class AlikedLightGlueExecuTorchIOAdapter(ExecuTorchIOAdapter):
    def __init__(self, model_name, model_wrapper=None, logger=None, config=None):
        super().__init__(model_name, model_wrapper, logger, config)
        self._input_shape = self._config.get("input_shape", (1, 3, 640, 640))

    def preprocess(self, inputs):
        image = inputs["image"]
        self._original_height, self._original_width = image.shape[:2]
        return super().preprocess(inputs)

    def postprocess(self, outputs):
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
            raise ValueError("Expected ALIKED outputs (keypoints, descriptors, scores)")
        keypoints, descriptors, scores = (torch.as_tensor(output) for output in outputs)
        keypoints = keypoints.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        keypoints[:, 0] *= self._original_width / self._input_shape[-1]
        keypoints[:, 1] *= self._original_height / self._input_shape[-2]
        return {
            "keypoints": np.asarray(cv.KeyPoint_convert(keypoints), dtype=object),
            "descriptors": descriptors.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False),
            "scores": scores.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False),
        }
