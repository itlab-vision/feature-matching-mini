import cv2 as cv
import numpy as np

from src.backend.io_adapter_base import IOAdapter


@IOAdapter.register("loftr")
class LoFTRIOAdapter(IOAdapter):
    def preprocess(self, inputs):
        if not isinstance(inputs, dict) or "image0" not in inputs or "image1" not in inputs:
            raise ValueError("LoFTR expects 'image0' and 'image1'")

        return {**inputs, "image0": np.asarray(inputs["image0"]),
                "image1": np.asarray(inputs["image1"])}

    def postprocess(self, outputs):
        result = dict(outputs)
        for key in ("keypoints0", "keypoints1"):
            points = np.asarray(result[key], dtype=np.float32).reshape(-1, 2)
            result[key] = np.asarray(cv.KeyPoint_convert(points), dtype=object)

        matches = np.asarray(result["matches"], dtype=np.int64).reshape(-1, 2)
        result["matches"] = np.asarray([cv.DMatch(int(i), int(j), 0) for i, j in matches])
        result["scores"] = np.asarray(result["scores"])
        return result
