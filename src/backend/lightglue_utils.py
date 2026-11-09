import cv2 as cv
import numpy as np


def keypoints_to_numpy(keypoints):
    if isinstance(keypoints, np.ndarray) and (keypoints.size == 0 or keypoints.dtype != object):
        return np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    keypoints = np.asarray(keypoints, dtype=object).reshape(-1)
    if keypoints.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    if hasattr(keypoints[0], "pt"):
        return np.asarray(cv.KeyPoint_convert(keypoints), dtype=np.float32).reshape(-1, 2)
    return np.asarray(keypoints.tolist(), dtype=np.float32).reshape(-1, 2)


def normalize_keypoints(keypoints, height, width):
    if height is None or width is None or height <= 0 or width <= 0:
        raise ValueError("LightGlue normalization requires positive image height and width")
    keypoints = keypoints_to_numpy(keypoints)
    size = np.asarray([width, height], dtype=np.float32)
    return (keypoints - size / 2) / (size.max() / 2)


def scale_and_orientation(features, num_keypoints):
    scales = features.get("scales")
    oris = features.get("oris")
    if scales is None:
        scales = np.ones(num_keypoints, dtype=np.float32)
    if oris is None:
        oris = np.zeros(num_keypoints, dtype=np.float32)
    if hasattr(scales, "detach"):
        scales = scales.detach().cpu().numpy()
    if hasattr(oris, "detach"):
        oris = oris.detach().cpu().numpy()
    scales = np.asarray(scales, dtype=np.float32).reshape(1, num_keypoints)
    oris = np.asarray(oris, dtype=np.float32).reshape(1, num_keypoints)
    return scales, oris
