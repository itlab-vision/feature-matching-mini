import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F

from src.backend.io_adapter_base import IOAdapter
from src.backend.executorch.io_adapter.io_adapter import ExecuTorchIOAdapter
from src.backend.executorch.utils import prepare_image_tensor


@IOAdapter.register("tfeat_executorch")
@IOAdapter.register("hardnet_executorch")
class PatchDescriptorExecuTorchIOAdapter(ExecuTorchIOAdapter):
    def __init__(self, model_name, model_wrapper=None, logger=None, config=None):
        super().__init__(model_name, model_wrapper, logger, config)

        self._patch_size = self._config.get("patch_size", 32)
        self._keypoints = np.empty((0,), dtype=object)
        self._scores = np.empty((0,), dtype=np.float32)

    def preprocess(self, inputs):
        if not isinstance(inputs, dict) or "image" not in inputs:
            raise ValueError("Patch descriptor adapter expects a dictionary containing 'image'")

        image = inputs["image"]
        keypoints = inputs.get("keypoints", inputs.get("kp"))
        if keypoints is None:
            raise ValueError("Patch descriptor adapter expects 'keypoints' or 'kp'")

        self._keypoints = np.asarray(keypoints, dtype=object)
        if len(self._keypoints) == 0:
            self._scores = np.empty((0,), dtype=np.float32)
            patches = torch.empty((0, 1, self._patch_size, self._patch_size), dtype=torch.float32)
            return {**inputs, "patches": patches}

        scores = inputs.get("scores")
        if scores is None:
            self._scores = np.ones((len(self._keypoints),), dtype=np.float32)
        else:
            self._scores = np.asarray(scores, dtype=np.float32)
            if self._scores.shape != (len(self._keypoints),):
                raise ValueError("Keypoint scores must have shape [N]")

        image = self._prepare_image(image)
        patches = self._extract_patches(image, self._keypoints, self._patch_size).contiguous()
        return {**inputs, "patches": patches}

    def postprocess(self, outputs):
        if isinstance(outputs, (tuple, list)):
            if len(outputs) != 1:
                raise ValueError("Expected one patch descriptor output tensor")
            outputs = outputs[0]

        if not isinstance(outputs, torch.Tensor) or outputs.ndim != 2:
            raise ValueError("Expected patch descriptors with shape [N, D]")
        if outputs.shape[0] != len(self._keypoints):
            raise ValueError("The number of descriptors does not match the number of keypoints")

        return {
            "keypoints": self._keypoints,
            "descriptors": outputs.detach().cpu().numpy().astype(np.float32, copy=False),
            "scores": self._scores,
        }

    @staticmethod
    def _prepare_image(image):
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                tensor = torch.from_numpy(np.ascontiguousarray(image)).float()[None, None] / 255.0
            elif image.ndim == 3 and image.shape[2] == 3:
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()[None] / 255.0
            elif image.ndim == 3 and image.shape[2] == 1:
                tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()[None] / 255.0
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        elif isinstance(image, torch.Tensor):
            tensor = prepare_image_tensor(image)
        else:
            raise TypeError("Patch descriptor adapter expects a numpy image or torch.Tensor")

        if tensor.shape[1] == 3:
            tensor = tensor[:, 0:1] * 0.2989 + tensor[:, 1:2] * 0.5870 + tensor[:, 2:3] * 0.1140
        if tensor.shape[1] != 1:
            raise ValueError(f"Expected one or three image channels, got {tensor.shape[1]}")
        return tensor.to(dtype=torch.float32, device="cpu")

    @staticmethod
    def _extract_patches(image, keypoints, patch_size):
        if hasattr(keypoints[0], "pt"):
            xy = torch.tensor([keypoint.pt for keypoint in keypoints], dtype=torch.float32)
        elif isinstance(keypoints, np.ndarray):
            xy = torch.from_numpy(np.asarray(keypoints, dtype=np.float32))
        elif isinstance(keypoints, torch.Tensor):
            xy = keypoints.to(dtype=torch.float32, device="cpu")
        else:
            xy = torch.tensor(keypoints, dtype=torch.float32)

        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("Keypoints must have shape [N, 2]")

        count = xy.shape[0]
        half = patch_size / 2.0
        coordinates = torch.linspace(-half + 0.5, half - 0.5, patch_size)
        grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        offsets = torch.stack((grid_x, grid_y), dim=-1)
        sample_pixels = xy.view(count, 1, 1, 2) + offsets

        _, _, height, width = image.shape
        sample_pixels[..., 0] = sample_pixels[..., 0] / max(width - 1, 1) * 2 - 1
        sample_pixels[..., 1] = sample_pixels[..., 1] / max(height - 1, 1) * 2 - 1

        expanded_image = image.expand(count, -1, -1, -1)
        return F.grid_sample(expanded_image, sample_pixels, align_corners=True)
