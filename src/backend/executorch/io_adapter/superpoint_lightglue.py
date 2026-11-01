import cv2 as cv
import numpy as np
import torch

from src.backend.io_adapter_base import IOAdapter
from src.backend.executorch.io_adapter.io_adapter import ExecuTorchIOAdapter
from src.backend.executorch.utils import nms_topk, sample_descriptors


@IOAdapter.register("superpoint_lightglue_executorch")
class SuperPointLightGlueExecuTorchIOAdapter(ExecuTorchIOAdapter):
    def __init__(self, model_name, model_wrapper=None, logger=None, config=None):
        super().__init__(model_name, model_wrapper, logger, config)

        self._input_shape = self._config.get("input_shape", (1, 3, 480, 640))
        self._num_keypoints = self._config.get("num_keypoints", 256)
        self._nms_radius = self._config.get("nms_radius", 4)
        self._original_height = None
        self._original_width = None
        self._input_height = None
        self._input_width = None

    def preprocess(self, inputs):
        if not isinstance(inputs, dict) or "image" not in inputs:
            raise ValueError("SuperPoint ExecuTorch adapter expects a dictionary containing 'image'")

        image = inputs["image"]
        if not isinstance(image, np.ndarray):
            raise TypeError("SuperPoint ExecuTorch adapter expects an OpenCV numpy image")

        processed = super().preprocess(inputs)
        self._original_height, self._original_width = image.shape[:2]
        self._input_height, self._input_width = processed["image"].shape[-2:]
        return processed

    def postprocess(self, outputs):
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
            raise ValueError("Expected SuperPoint outputs (logits, probabilities, dense_descriptors)")

        _, probabilities, dense = outputs
        if (not isinstance(probabilities, torch.Tensor) or probabilities.ndim != 4
                or probabilities.shape[0] != 1):
            raise ValueError("Expected SuperPoint probabilities with shape [1, 65, H, W]")
        if probabilities.shape[1] != 65:
            raise ValueError(f"Expected 65 SuperPoint probability channels, got {probabilities.shape[1]}")
        if not isinstance(dense, torch.Tensor) or dense.ndim != 4 or dense.shape[0] != 1:
            raise ValueError("Expected SuperPoint descriptors with shape [1, C, H, W]")
        if dense.shape[-2:] != probabilities.shape[-2:]:
            raise ValueError("SuperPoint probabilities and descriptors must have the same spatial shape")

        probabilities = probabilities[:, :-1]
        _, _, coarse_height, coarse_width = probabilities.shape
        scores = probabilities.permute(0, 2, 3, 1).reshape(
            1, coarse_height, coarse_width, 8, 8
        )
        scores = scores.permute(0, 1, 3, 2, 4).reshape(
            coarse_height * 8, coarse_width * 8
        )

        keypoints, values = nms_topk(scores, self._num_keypoints, self._nms_radius)
        descriptors = sample_descriptors(dense, keypoints / 8, coarse_height, coarse_width)

        input_height = self._input_height or coarse_height * 8
        input_width = self._input_width or coarse_width * 8
        if self._original_width is not None and self._original_height is not None:
            keypoints[:, 0] *= self._original_width / input_width
            keypoints[:, 1] *= self._original_height / input_height

        keypoints_np = keypoints.detach().cpu().numpy().astype(np.float32, copy=False)
        keypoints_cv = np.asarray(cv.KeyPoint_convert(keypoints_np), dtype=object)

        return {
            "keypoints": keypoints_cv,
            "descriptors": descriptors.detach().cpu().numpy().astype(np.float32, copy=False),
            "scores": values.detach().cpu().numpy().astype(np.float32, copy=False),
        }
