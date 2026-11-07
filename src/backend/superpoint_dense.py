import cv2 as cv
import numpy as np
import torch

from src.backend.executorch.utils import nms_topk, sample_descriptors


def postprocess_superpoint_dense(outputs, original_size, input_size, num_keypoints, nms_radius=4):
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
        raise ValueError("Expected SuperPoint outputs (logits, probabilities, dense_descriptors)")

    probabilities = torch.as_tensor(np.ascontiguousarray(outputs[1]))
    dense = torch.as_tensor(np.ascontiguousarray(outputs[2]))
    if probabilities.ndim != 4 or probabilities.shape[:2] != (1, 65):
        raise ValueError("Expected SuperPoint probabilities with shape [1, 65, H, W]")
    if dense.ndim != 4 or dense.shape[0] != 1 or dense.shape[-2:] != probabilities.shape[-2:]:
        raise ValueError("Expected SuperPoint descriptors with shape [1, C, H, W]")

    probabilities = probabilities[:, :-1]
    coarse_height, coarse_width = probabilities.shape[-2:]
    scores = probabilities.permute(0, 2, 3, 1).reshape(1, coarse_height, coarse_width, 8, 8)
    scores = scores.permute(0, 1, 3, 2, 4).reshape(coarse_height * 8, coarse_width * 8)

    keypoints, values = nms_topk(scores, num_keypoints, nms_radius)
    descriptors = sample_descriptors(dense, keypoints / 8, coarse_height, coarse_width)

    original_height, original_width = original_size
    input_height, input_width = input_size
    keypoints[:, 0] *= original_width / input_width
    keypoints[:, 1] *= original_height / input_height

    keypoints_np = keypoints.detach().cpu().numpy().astype(np.float32, copy=False)
    keypoints_cv = np.asarray(cv.KeyPoint_convert(keypoints_np), dtype=object)
    return {
        "keypoints": keypoints_cv,
        "descriptors": descriptors.detach().cpu().numpy().astype(np.float32, copy=False),
        "scores": values.detach().cpu().numpy().astype(np.float32, copy=False),
    }
