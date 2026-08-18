import cv2 as cv
import numpy as np

from feature_matching.preprocessor import Preprocessor
from feature_matching.algorithms import (DETECTOR_DESCRIPTOR_COMPATIBILITY, DESCRIPTOR_MATCHER_COMPATIBILITY,
                        DNN_MATCHERS, OPENCV_MATCHERS)

from feature_matching.detectors import Detector
from feature_matching.descriptors import Descriptor
from feature_matching.matchers import Matcher


class FeatureMatcherCV2:
    _DETECTOR_DESCRIPTOR_COMPATIBILITY = DETECTOR_DESCRIPTOR_COMPATIBILITY

    def __init__(self, logger, detector='sift', descriptor='sift', matcher='bf', config=None):
        if config is None:
            config = {}

        self._detector = detector
        self._descriptor = descriptor
        self._matcher = matcher
        self._logger = logger

        self._detector_config = config.get('detector', {})
        self._descriptor_config = config.get('descriptor', {})
        self._matcher_config = config.get('matcher', {})
        self._preprocessor_config = config.get('preprocessor', {})

        self._validate_compatibility()

    def _validate_compatibility(self):
        if self._detector not in DETECTOR_DESCRIPTOR_COMPATIBILITY:
            raise ValueError(f"Detector '{self._detector}' is not registered in compatibility matrix")

        if self._descriptor not in DETECTOR_DESCRIPTOR_COMPATIBILITY[self._detector]:
            raise ValueError(f"Detector {self._detector} cannot be used with Descriptor {self._descriptor}")

        if self._descriptor in DESCRIPTOR_MATCHER_COMPATIBILITY:
            if self._matcher not in DESCRIPTOR_MATCHER_COMPATIBILITY[self._descriptor]:
                raise ValueError(f"Descriptor '{self._descriptor}' cannot be used with Matcher '{self._matcher}'."
                                 f" Available: {DESCRIPTOR_MATCHER_COMPATIBILITY[self._descriptor]}")

        if self._matcher in DNN_MATCHERS and 'mode' in self._matcher_config:
            raise ValueError(f"Matcher '{self._matcher}' does not support 'mode' parameter. "
                             f"Mode is only available for OpenCV matchers: {OPENCV_MATCHERS}")

    def _has_keypoints(self, features):
        kp = features.get('kp') or features.get('keypoints')
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

        if not correspondences.get('matches') or len(correspondences.get('matches')) == 0:
            self._logger.warning("No matches found to visualize.")
            return cv.drawMatches(img0, features0.get('kp'), img1, features1.get('kp'),
                                  [], None, **draw_params)

        mode = self._matcher_config.get('mode', 'simple')

        if mode == 'simple':
            return cv.drawMatches(img0, features0.get('kp'), img1, features1.get('kp'),
                                  correspondences.get('matches'), None, **draw_params)
        if mode == 'knn':
            return cv.drawMatchesKnn(img0, features0.get('kp'), img1, features1.get('kp'),
                                     correspondences.get('matches'), None, **draw_params)

        return cv.drawMatches(img0, features0.get('kp'), img1, features1.get('kp'),
                              [], None, **draw_params)

    def match(self, img0, img1):
        detector = Detector.create(detector_name=self._detector, logger=self._logger, config=self._detector_config)
        descriptor = Descriptor.create(descriptor_name=self._descriptor, logger=self._logger,
                                       config=self._descriptor_config)
        matcher = Matcher.create(matcher_name=self._matcher, descriptor_name=descriptor,
                                 logger=self._logger, config=self._matcher_config)
        preprocessor = Preprocessor(config=self._preprocessor_config, logger=self._logger)

        img0 = preprocessor.prepare_image(img0, from_algo='opencv', to_algo=self._detector)
        img1 = preprocessor.prepare_image(img1, from_algo='opencv', to_algo=self._detector)

        features0 = detector.detect(img0)
        features0 = preprocessor.prepare_features(features0, from_algo=self._detector, to_algo=self._descriptor)
        features0 = descriptor.compute(img0, features0)

        features1 = detector.detect(img1)
        features1 = preprocessor.prepare_features(features1, from_algo=self._detector, to_algo=self._descriptor)
        features1 = descriptor.compute(img1, features1)

        if not self._has_keypoints(features0) or not self._has_keypoints(features1):
            raise ValueError("Failed to detect key points")

        features0 = preprocessor.prepare_features(features0, from_algo=self._descriptor, to_algo=self._matcher)
        features1 = preprocessor.prepare_features(features1, from_algo=self._descriptor, to_algo=self._matcher)

        correspondences = matcher.match(features0, features1)
        correspondences = preprocessor.prepare_matches(correspondences, from_algo=self._matcher, to_algo='opencv')

        features0 = preprocessor.prepare_features(features0, from_algo=self._matcher, to_algo='opencv')
        features1 = preprocessor.prepare_features(features1, from_algo=self._matcher, to_algo='opencv')

        return features0, features1, correspondences
