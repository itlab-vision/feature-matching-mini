import cv2 as cv
import numpy as np
from src.backend.io_adapter_base import IOAdapter
from src.backend.opencv.io_adapter.io_adapter import OpenCVIOAdapter


@IOAdapter.register("aliked_opencv")
class AlikedOpenCVIOAdapter(OpenCVIOAdapter):
    def preprocess(self, inputs):
        img = inputs.pop('image')
        self.orig_h, self.orig_w = img.shape[:2]
        blob = cv.dnn.blobFromImage(img, scalefactor=1.0 / 255.0, size=(640, 640), swapRB=True, crop=False)
        self._model.setInput(blob)

    def postprocess(self, outputs):
        keypoints = outputs[0]
        descriptors = outputs[1]
        scores = outputs[2]
        keypoints_np = keypoints.astype(np.float32)
        if keypoints_np.ndim == 3:
            keypoints_np = keypoints_np.reshape(-1, 2)

        keypoints_np = (keypoints_np + 1.0) / 2.0

        keypoints_np[:, 0] *= self.orig_w
        keypoints_np[:, 1] *= self.orig_h

        keypoints = cv.KeyPoint_convert(keypoints_np)
        keypoints = np.asarray(keypoints)

        result = {'keypoints': keypoints, 'descriptors': descriptors, 'scores': scores}
        return result
