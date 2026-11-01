import cv2 as cv
import numpy as np
import torch

from src.backend.io_adapter_base import IOAdapter
from src.backend.executorch.io_adapter.io_adapter import ExecuTorchIOAdapter
from src.backend.executorch.utils import nms_topk, sample_descriptors


@IOAdapter.register("disk_lightglue_executorch")
class DiskLightGlueExecuTorchIOAdapter(ExecuTorchIOAdapter):
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
            raise ValueError("DISK ExecuTorch adapter expects a dictionary containing 'image'")

        image = inputs["image"]
        if not isinstance(image, np.ndarray):
            raise TypeError("DISK ExecuTorch adapter expects an OpenCV numpy image")

        processed = super().preprocess(inputs)
        processed["image"].mul_(255.0)
        self._original_height, self._original_width = image.shape[:2]
        self._input_height, self._input_width = processed["image"].shape[-2:]
        return processed

    def postprocess(self, outputs):
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
            num_outputs = len(outputs) if isinstance(outputs, (tuple, list)) else 1
            raise ValueError(f"DISK ExecuTorch model must return heatmap and dense descriptors,"
                             f" got {num_outputs} output(s)")

        heatmap, dense = outputs
        if not isinstance(heatmap, torch.Tensor) or heatmap.ndim != 4 or heatmap.shape[:2] != (1, 1):
            raise ValueError("Expected DISK heatmap with shape [1, 1, H, W]")
        if not isinstance(dense, torch.Tensor) or dense.ndim != 4 or dense.shape[0] != 1:
            raise ValueError("Expected DISK descriptors with shape [1, C, H, W]")
        if dense.shape[-2:] != heatmap.shape[-2:]:
            raise ValueError("DISK heatmap and descriptors must have the same spatial shape")

        feature_height, feature_width = heatmap.shape[-2:]
        keypoints, scores = nms_topk(heatmap[0, 0], self._num_keypoints, self._nms_radius)
        descriptors = sample_descriptors(dense, keypoints, feature_height, feature_width)

        input_height = self._input_height or feature_height
        input_width = self._input_width or feature_width
        if self._original_width is not None and self._original_height is not None:
            keypoints[:, 0] *= self._original_width / input_width
            keypoints[:, 1] *= self._original_height / input_height

        keypoints_np = keypoints.detach().cpu().numpy().astype(np.float32, copy=False)
        keypoints_cv = np.asarray(cv.KeyPoint_convert(keypoints_np), dtype=object)

        return {
            "keypoints": keypoints_cv,
            "descriptors": descriptors.detach().cpu().numpy().astype(np.float32, copy=False),
            "scores": scores.detach().cpu().numpy().astype(np.float32, copy=False),
        }
