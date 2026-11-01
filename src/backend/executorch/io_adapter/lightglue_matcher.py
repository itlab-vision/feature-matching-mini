import cv2 as cv
import numpy as np
import torch

from src.backend.io_adapter_base import IOAdapter
from src.backend.executorch.io_adapter.io_adapter import ExecuTorchIOAdapter


@IOAdapter.register("lightglue_disk_executorch")
@IOAdapter.register("lightglue_superpoint_executorch")
@IOAdapter.register("lightglue_executorch")
class LightGlueExecuTorchIOAdapter(ExecuTorchIOAdapter):
    def __init__(self, model_name, model_wrapper=None, logger=None, config=None):
        super().__init__(model_name, model_wrapper, logger, config)
        self._num_keypoints = self._config.get("num_keypoints", 256)

    def preprocess(self, inputs):
        if not isinstance(inputs, dict):
            raise TypeError("LightGlue ExecuTorch adapter expects a dictionary")

        features0 = self._prepare_features(inputs.get("features0"), "features0")
        features1 = self._prepare_features(inputs.get("features1"), "features1")
        return {
            "keypoints0": features0["keypoints"],
            "keypoints1": features1["keypoints"],
            "descriptors0": features0["descriptors"],
            "descriptors1": features1["descriptors"],
        }

    def postprocess(self, outputs):
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 4:
            raise ValueError("Expected LightGlue outputs (matches0, matches1, scores0, scores1)")

        matches0, _, scores0, _ = outputs
        if not isinstance(matches0, torch.Tensor) or matches0.shape != (1, self._num_keypoints):
            raise ValueError(f"Expected matches0 with shape [1, {self._num_keypoints}]")
        if not isinstance(scores0, torch.Tensor) or scores0.shape != (1, self._num_keypoints):
            raise ValueError(f"Expected scores0 with shape [1, {self._num_keypoints}]")

        valid = (matches0[0] >= 0) & (matches0[0] < self._num_keypoints)
        query_indices = torch.arange(self._num_keypoints, dtype=torch.long)[valid]
        train_indices = matches0[0][valid].to(torch.long)
        match_scores = scores0[0][valid]

        matches = []
        for query_index, train_index in zip(query_indices.tolist(), train_indices.tolist()):
            match = cv.DMatch()
            match.queryIdx = query_index
            match.trainIdx = train_index
            matches.append(match)

        return {
            "matches": np.asarray(matches, dtype=object),
            "scores": match_scores.detach().cpu().numpy().astype(np.float32, copy=False),
        }

    def _prepare_features(self, features, name):
        if not isinstance(features, dict):
            raise ValueError(f"LightGlue ExecuTorch adapter expects '{name}'")

        keypoints = features.get("keypoints", features.get("kp"))
        descriptors = features.get("descriptors", features.get("des"))
        if keypoints is None or descriptors is None:
            raise ValueError(f"{name} must contain keypoints and descriptors")

        keypoints = self._keypoints_to_tensor(keypoints)
        descriptors = self._descriptors_to_tensor(descriptors)
        if keypoints.shape[0] != self._num_keypoints:
            raise ValueError(
                f"LightGlue ExecuTorch model expects {self._num_keypoints} keypoints, "
                f"but {name} contains {keypoints.shape[0]}"
            )
        if descriptors.shape[0] != self._num_keypoints:
            raise ValueError(
                f"LightGlue ExecuTorch model expects {self._num_keypoints} descriptors, "
                f"but {name} contains {descriptors.shape[0]}"
            )

        return {
            "keypoints": keypoints.unsqueeze(0).contiguous(),
            "descriptors": descriptors.unsqueeze(0).contiguous(),
        }

    @staticmethod
    def _keypoints_to_tensor(keypoints):
        if isinstance(keypoints, torch.Tensor):
            tensor = keypoints.detach().to(dtype=torch.float32, device="cpu")
        else:
            keypoints = np.asarray(keypoints)
            if keypoints.size == 0:
                tensor = torch.empty((0, 2), dtype=torch.float32)
            elif hasattr(keypoints.reshape(-1)[0], "pt"):
                coordinates = [keypoint.pt for keypoint in keypoints.reshape(-1)]
                tensor = torch.tensor(coordinates, dtype=torch.float32)
            else:
                tensor = torch.from_numpy(np.asarray(keypoints, dtype=np.float32))

        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim != 2 or tensor.shape[1] != 2:
            raise ValueError("LightGlue keypoints must have shape [N, 2]")
        return tensor

    @staticmethod
    def _descriptors_to_tensor(descriptors):
        if isinstance(descriptors, torch.Tensor):
            tensor = descriptors.detach().to(dtype=torch.float32, device="cpu")
        else:
            tensor = torch.from_numpy(np.ascontiguousarray(descriptors, dtype=np.float32))

        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim != 2:
            raise ValueError("LightGlue descriptors must have shape [N, D]")
        return tensor
