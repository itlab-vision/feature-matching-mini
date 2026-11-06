import cv2 as cv
import numpy as np

from src.backend.io_adapter_base import IOAdapter


@IOAdapter.register("r2d2")
class R2D2IOAdapter(IOAdapter):
    def preprocess(self, inputs):
        if not isinstance(inputs, dict) or "image" not in inputs:
            raise ValueError("R2D2 expects a dictionary containing 'image'")

        image = np.asarray(inputs["image"])
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)

        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Unsupported R2D2 image shape: {image.shape}")
        return {**inputs, "image": np.ascontiguousarray(image)}

    def postprocess(self, outputs):
        result = dict(outputs)
        points = np.asarray(result["keypoints"], dtype=np.float32).reshape(-1, 2)
        result["keypoints"] = np.asarray(cv.KeyPoint_convert(points), dtype=object)
        result["descriptors"] = np.asarray(result["descriptors"])
        result["scores"] = np.asarray(result["scores"])
        return result
