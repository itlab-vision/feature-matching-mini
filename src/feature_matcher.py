import cv2 as cv
import numpy as np

from src.algorithms import (DETECTOR_DESCRIPTOR_COMPATIBILITY, DESCRIPTOR_MATCHER_COMPATIBILITY, DNN_MATCHERS,
                            OPENCV_MATCHERS, DNN_PIPELINES)

from src.detectors import Detector
from src.descriptors import Descriptor
from src.matchers import Matcher
from src.dnn_extractors import DNNFeatureExtractors  # noqa: F401
from src.dnn_matchers import DNNMatcher  # noqa: F401
from src.dnn_pipeline import DNNPipeline  # noqa: F401
from src.opencv_dnn_extractors import ALIKEDOpenCV, DISKOpenCV  # noqa: F401
from src.opencv_dnn_matchers import LightGlueOpenCVMatcher  # noqa: F401
from src.hardnet_descriptor import HardNet  # noqa: F401
from src.tfeat_descriptor import TFeat  # noqa: F401


class FeatureMatcherCV2:
    _DETECTOR_DESCRIPTOR_COMPATIBILITY = DETECTOR_DESCRIPTOR_COMPATIBILITY

    def __init__(self, logger, detector='sift', descriptor='sift', matcher='bf', config=None):
        if config is None:
            config = {}

        self._detector_name = detector
        self._descriptor_name = descriptor
        self._matcher_name = matcher
        self._logger = logger

        self._detector_config = config.get('detector', {})
        self._descriptor_config = config.get('descriptor', {})
        self._matcher_config = config.get('matcher', {})

        self._validate_compatibility()

        self._detector_class = Detector.create(detector_name=self._detector_name, logger=self._logger,
                                               config=self._detector_config)
        self._descriptor_class = Descriptor.create(descriptor_name=self._descriptor_name, logger=self._logger,
                                                   config=self._descriptor_config)
        self._matcher_class = Matcher.create(matcher_name=self._matcher_name, descriptor_class=self._descriptor_class,
                                             logger=self._logger, config=self._matcher_config)

    def _validate_compatibility(self):
        if self._detector_name not in DETECTOR_DESCRIPTOR_COMPATIBILITY:
            raise ValueError(f"Detector '{self._detector_name}' is not registered in compatibility matrix")

        if self._descriptor_name not in DETECTOR_DESCRIPTOR_COMPATIBILITY[self._detector_name]:
            raise ValueError(f"Detector {self._detector_name} cannot be used with Descriptor {self._descriptor_name}")

        if self._descriptor_name in DESCRIPTOR_MATCHER_COMPATIBILITY:
            if self._matcher_name not in DESCRIPTOR_MATCHER_COMPATIBILITY[self._descriptor_name]:
                raise ValueError(f"Descriptor '{self._descriptor_name}' cannot be used with "
                                 f"Matcher '{self._matcher_name}'"
                                 f"Available: {DESCRIPTOR_MATCHER_COMPATIBILITY[self._descriptor_name]}")

        if self._matcher_name in DNN_MATCHERS and 'mode' in self._matcher_config:
            raise ValueError(f"Matcher '{self._matcher_name}' does not support 'mode' parameter. "
                             f"Mode is only available for OpenCV matchers: {OPENCV_MATCHERS}")

    def _has_keypoints(self, features):
        if self._detector_name in DNN_PIPELINES:
            return True

        kp = features.get('kp')
        if kp is None:
            kp = features.get('keypoints')

        if kp is None:
            return False
        if isinstance(kp, (list, tuple)):
            return len(kp) > 0
        if isinstance(kp, np.ndarray):
            return kp.size > 0
        return True

    def visualize_matches(self, img0, features0, img1, features1, correspondences):
        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(0, 0, 255),
                           flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        matches = correspondences.get('matches')
        if matches is None or len(matches) == 0:
            self._logger.warning("No matches found to visualize.")
            return cv.drawMatches(img0, features0.get('keypoints'), img1, features1.get('keypoints'),
                                  [], None, **draw_params)

        if 'keypoints0' in correspondences and 'keypoints1' in correspondences:
            kp0 = correspondences['keypoints0']
            kp1 = correspondences['keypoints1']
        elif 'keypoints' in features0 and 'keypoints' in features1:
            kp0 = features0.get('keypoints')
            kp1 = features1.get('keypoints')
        elif 'kp' in features0 and 'kp' in features1:
            kp0 = features0.get('kp')
            kp1 = features1.get('kp')
        else:
            self._logger.warning("Keypoints are missing, cannot visualize matches")
            return cv.drawMatches(img0, [], img1, [], [], None, **draw_params)

        mode = self._matcher_config.get('mode', 'simple')
        if mode == 'simple':
            return cv.drawMatches(img0, kp0, img1, kp1, matches, None, **draw_params)
        if mode == 'knn':
            return cv.drawMatchesKnn(img0, kp0, img1, kp1, matches, None, **draw_params)

        return cv.drawMatches(img0, kp0, img1, kp1, [], None, **draw_params)

    def match(self, img0, img1):
        features0 = self._detector_class.detect(img0)
        features0 = self._descriptor_class.compute(img0, features0)

        features1 = self._detector_class.detect(img1)
        features1 = self._descriptor_class.compute(img1, features1)

        if not self._has_keypoints(features0) or not self._has_keypoints(features1):
            self._logger.warning("Failed to detect key points")

        correspondences = self._matcher_class.match(features0, features1)
        return features0, features1, correspondences

    def close(self):
        objects = (self._matcher_class, self._descriptor_class, self._detector_class)

        for obj in objects:
            if obj is not None and hasattr(obj, "close"):
                obj.close()

        self._matcher_class = None
        self._descriptor_class = None
        self._detector_class = None
