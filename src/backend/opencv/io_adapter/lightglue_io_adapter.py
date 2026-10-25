import cv2 as cv
import numpy as np
from src.backend.io_adapter_base import IOAdapter
from src.backend.opencv.io_adapter.io_adapter import OpenCVIOAdapter


@IOAdapter.register("lightglue_opencv")
class LightGlueOpenCVIOAdapter(OpenCVIOAdapter):
    def _normalize_keypoints(self, keypoints, img_shape, extractor_name):
        keypoints = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2).copy()
        height, width = img_shape[:2]

        if "disk" in extractor_name:
            if width <= 1 or height <= 1:
                raise ValueError("Width and height must be more than 1")
            keypoints[:, 0] /= width - 1
            keypoints[:, 1] /= height - 1

        elif "aliked" in extractor_name:
            if width <= 1 or height <= 1:
                raise ValueError("Width and height must be more than 1")
            keypoints[:, 0] = 2.0 * keypoints[:, 0] / width - 1.0
            keypoints[:, 1] = 2.0 * keypoints[:, 1] / height - 1.0

        return keypoints

    def preprocess(self, inputs):
        features1, features2 = inputs.get('features1'), inputs.get('features2')
        kpts0 = features1.get('keypoints')
        kpts1 = features2.get('keypoints')
        desc0 = features1.get('descriptors')
        desc1 = features2.get('descriptors')

        if kpts0 is None or kpts1 is None or len(kpts0) == 0 or len(kpts1) == 0 or \
                desc0 is None or desc1 is None or len(desc0) == 0 or len(desc1) == 0:
            return None

        shape0 = features1.get("img_shape")
        shape1 = features2.get("img_shape")

        extractor_name = features1.get("extractor_name", "").lower()

        if len(kpts0) > 0 and isinstance(kpts0[0], cv.KeyPoint):
            kpts0 = cv.KeyPoint_convert(kpts0)
        if len(kpts1) > 0 and isinstance(kpts1[0], cv.KeyPoint):
            kpts1 = cv.KeyPoint_convert(kpts1)

        kpts0 = self._normalize_keypoints(kpts0, shape0, extractor_name)
        kpts1 = self._normalize_keypoints(kpts1, shape1, extractor_name)

        if kpts0.ndim == 2:
            kpts0 = np.expand_dims(kpts0, axis=0)
        if kpts1.ndim == 2:
            kpts1 = np.expand_dims(kpts1, axis=0)
        if desc0.ndim == 2:
            desc0 = np.expand_dims(desc0, axis=0)
        if desc1.ndim == 2:
            desc1 = np.expand_dims(desc1, axis=0)

        input_layers = {
            'kpts0': kpts0.astype(np.float32),
            'kpts1': kpts1.astype(np.float32),
            'desc0': desc0.astype(np.float32),
            'desc1': desc1.astype(np.float32)
        }
        for name, value in input_layers.items():
            self._model.setInput(value, name=name)

        return input_layers

    def postprocess(self, outputs):
        if outputs is None:
            return {
                'matches': [],
                'scores': []
            }
        matches = outputs[0]
        if matches.ndim > 2 and matches.shape[0] == 1:
            matches = np.squeeze(matches, axis=0)

        scores = outputs[1] if len(outputs) > 1 else None
        if scores is not None and scores.ndim > 1 and scores.shape[0] == 1:
            scores = np.squeeze(scores, axis=0)

        if scores is not None:
            mask = scores > self.scoreThreshold
            matches = matches[mask]
            scores = scores[mask]

        dmatches = []
        for i, match in enumerate(matches):
            query_idx, target_idx = int(match[0]), int(match[1])
            score = float(scores[i]) if scores is not None and i < len(scores) else 1.0
            distance = 1.0 - score if score <= 1.0 else 0.0
            dmatches.append(cv.DMatch(_queryIdx=query_idx, _trainIdx=target_idx, _imgIdx=0, _distance=distance))

        result = {'matches': dmatches, 'scores': scores}
        return result
