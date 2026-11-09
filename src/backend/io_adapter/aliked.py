import cv2 as cv
import numpy as np

from src.backend.io_adapter_base import IOAdapter


@IOAdapter.register('aliked')
class ALIKEDIOAdapter(IOAdapter):
    def preprocess(self, inputs):
        if not isinstance(inputs, dict) or 'image' not in inputs:
            raise ValueError("ALIKED expects a dictionary containing 'image'")
        image = np.asarray(inputs['image'])
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f'Unsupported ALIKED image shape: {image.shape}')
        self._image_size = image.shape[:2]
        return {**inputs, 'image': np.ascontiguousarray(image)}

    def postprocess(self, outputs):
        keypoints, descriptors, scores = outputs
        points = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2).copy()
        points = (points + 1.0) / 2.0
        points[:, 0] *= self._image_size[1]
        points[:, 1] *= self._image_size[0]
        return {'keypoints': np.asarray(cv.KeyPoint_convert(points), dtype=object),
                'descriptors': np.asarray(descriptors).reshape(-1, np.asarray(descriptors).shape[-1]),
                'scores': np.asarray(scores).reshape(-1)}
