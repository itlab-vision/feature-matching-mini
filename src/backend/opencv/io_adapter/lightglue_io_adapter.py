import cv2 as cv
import numpy as np

from src.backend.io_adapter_base import IOAdapter
from src.backend.lightglue_utils import normalize_keypoints, scale_and_orientation
from src.backend.opencv.io_adapter.io_adapter import OpenCVIOAdapter


@IOAdapter.register("lightglue_opencv")
class LightGlueOpenCVIOAdapter(OpenCVIOAdapter):
    def __init__(self, model_name, model_wrapper=None, logger=None, config=None):
        super().__init__(model_name, model_wrapper, logger, config)
        self._normalize = self._config.get("normalize_keypoints", False)
        descriptor_class = self._config.get("descriptor_class")
        descriptor_name = descriptor_class._descriptor_name if descriptor_class is not None else ""
        self._add_scale_ori = "sift" in descriptor_name or "doghardnet" in descriptor_name

    def _normalize_keypoints(self, keypoints, height, width, extractor_name):
        keypoints = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2).copy()
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

    def _prepare_keypoints(self, features, extractor_name):
        keypoints = features.get("keypoints")
        if self._normalize:
            return normalize_keypoints(keypoints, features.get("height"), features.get("width"))
        if len(keypoints) > 0 and isinstance(keypoints[0], cv.KeyPoint):
            keypoints = cv.KeyPoint_convert(keypoints)
        return self._normalize_keypoints(keypoints, features.get("height"), features.get("width"), extractor_name)

    def preprocess(self, inputs):
        features0, features1 = inputs.get("features0"), inputs.get("features1")
        if not features0 or not features1:
            return None
        desc0 = features0.get("descriptors")
        desc1 = features1.get("descriptors")
        if len(features0.get("keypoints", ())) == 0 or len(features1.get("keypoints", ())) == 0 or \
                desc0 is None or desc1 is None or len(desc0) == 0 or len(desc1) == 0:
            return None

        extractor_name = features0.get("extractor_name", "").lower()
        kpts0 = self._prepare_keypoints(features0, extractor_name)[None]
        kpts1 = self._prepare_keypoints(features1, extractor_name)[None]
        input_layers = {
            "kpts0": np.ascontiguousarray(kpts0, dtype=np.float32),
            "kpts1": np.ascontiguousarray(kpts1, dtype=np.float32),
            "desc0": np.ascontiguousarray(desc0[None], dtype=np.float32),
            "desc1": np.ascontiguousarray(desc1[None], dtype=np.float32),
        }
        if self._add_scale_ori:
            scales0, oris0 = scale_and_orientation(features0, kpts0.shape[1])
            scales1, oris1 = scale_and_orientation(features1, kpts1.shape[1])
            input_layers.update({"scales0": scales0, "oris0": oris0, "scales1": scales1, "oris1": oris1})
        for name, value in input_layers.items():
            self._model.setInput(value, name=name)
        return input_layers

    def postprocess(self, outputs):
        if outputs is None:
            return {"matches": [], "scores": []}
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
            score = float(scores[i]) if scores is not None and i < len(scores) else 1.0
            distance = 1.0 - score if score <= 1.0 else 0.0
            dmatches.append(cv.DMatch(_queryIdx=int(match[0]), _trainIdx=int(match[1]),
                                      _imgIdx=0, _distance=distance))
        return {"matches": dmatches, "scores": scores}
