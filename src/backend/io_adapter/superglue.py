import cv2 as cv
import numpy as np

from src.backend.io_adapter_base import IOAdapter


@IOAdapter.register("superglue")
class SuperGlueIOAdapter(IOAdapter):
    def preprocess(self, inputs):
        if not isinstance(inputs, dict) or "features0" not in inputs or "features1" not in inputs:
            raise ValueError("SuperGlue expects 'features0' and 'features1'")

        result = dict(inputs)
        for key in ("features0", "features1"):
            features = dict(inputs[key])
            points = features["keypoints"]

            if len(points) and isinstance(points[0], cv.KeyPoint):
                points = cv.KeyPoint_convert(points)

            features["keypoints"] = np.asarray(points, dtype=np.float32).reshape(-1, 2)
            features["descriptors"] = np.asarray(features["descriptors"], dtype=np.float32)
            features["scores"] = np.asarray(features["scores"], dtype=np.float32)
            result[key] = features
        return result

    def postprocess(self, outputs):
        result = dict(outputs)
        matches = np.asarray(result["matches"], dtype=np.int64).reshape(-1, 2)
        result["matches"] = np.asarray([cv.DMatch(int(i), int(j), 0) for i, j in matches])
        result["scores"] = np.asarray(result["scores"])
        return result
