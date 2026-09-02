import cv2 as cv
from abc import ABC, abstractmethod

from src.algorithms import ALL_DETECTORS


class Detector(ABC):
    _METHODS = {}

    def __init__(self, logger, detector_name='sift'):
        self._detector_name = detector_name
        self._logger = logger

    def __init_subclass__(cls, register=True, **kwargs):
        super().__init_subclass__(**kwargs)

        if hasattr(cls, '_EXTRACTOR_CLASSES'):
            for key in cls._EXTRACTOR_CLASSES.keys():
                Detector._METHODS[key] = cls

        elif register:
            key = cls.__name__.replace("Detector", "").lower()

            for key_ in ALL_DETECTORS:
                if key_.replace("_", "") == key:
                    key = key_
                    break

            if key:
                Detector._METHODS[key] = cls

    @staticmethod
    def create(detector_name, logger, config=None):
        if config is None:
            config = {}

        detector_class_name = detector_name.lower()
        if detector_class_name not in Detector._METHODS:
            raise ValueError(f"Detector '{detector_class_name}' not found."
                             f" Available: {list(Detector._METHODS.keys())}")

        return Detector._METHODS[detector_class_name](detector_class_name, logger, config)

    @abstractmethod
    def detect(self, img):
        pass


class OpenCVDetector(Detector, register=False):
    def __init__(self, detector_name, logger, extractor, config):
        super().__init__(logger, detector_name)
        self._extractor = extractor
        self._nfeatures = config.get('nfeatures')

    def detect(self, img):
        if img is None:
            self._logger.error("Input image is None. Detection aborted.")
            return {'kp': ()}

        self._logger.info(f"Detecting keypoints with {self._detector_name}")
        kp = self._extractor.detect(img, None)

        if kp and self._nfeatures is not None and len(kp) > self._nfeatures:
            kp = sorted(kp, key=lambda x: x.response, reverse=True)[:self._nfeatures]

        if kp:
            self._logger.info(f"{self._detector_name} found {len(kp)} points")
        else:
            self._logger.warning(f"{self._detector_name} found 0 points")
        return {'kp': kp}


class SIFTDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, config):
        cv_params = {}
        if 'nfeatures' in config:
            cv_params['nfeatures'] = config['nfeatures']
        if 'nOctaveLayers' in config:
            cv_params['nOctaveLayers'] = config['nOctaveLayers']
        super().__init__(detector_name, logger, cv.SIFT_create(**cv_params), config)


class ORBDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, config):
        cv_params = {}
        if 'nfeatures' in config:
            cv_params['nfeatures'] = config['nfeatures']
        if 'threshold' in config:
            cv_params['fastThreshold'] = int(config['threshold'])
        super().__init__(detector_name, logger, cv.ORB_create(**cv_params), config)


class FASTDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, config):
        cv_params = {}
        if 'threshold' in config:
            cv_params['threshold'] = int(config['threshold'])
        super().__init__(detector_name, logger, cv.FastFeatureDetector_create(**cv_params), config)


class AKAZEDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, config):
        cv_params = {}
        if 'threshold' in config:
            cv_params['threshold'] = float(config['threshold'])
        if 'nOctaveLayers' in config:
            cv_params['nOctaveLayers'] = config['nOctaveLayers']
        super().__init__(detector_name, logger, cv.xfeatures2d.AKAZE_create(**cv_params), config)


class BRISKDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, config):
        cv_params = {}
        if 'threshold' in config:
            cv_params['thresh'] = int(config['threshold'])
        super().__init__(detector_name, logger, cv.xfeatures2d.BRISK_create(**cv_params), config)


class KAZEDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, config):
        cv_params = {}
        if 'threshold' in config:
            cv_params['threshold'] = float(config['threshold'])
        if 'nOctaveLayers' in config:
            cv_params['nOctaveLayers'] = config['nOctaveLayers']
        super().__init__(detector_name, logger, cv.xfeatures2d.KAZE_create(**cv_params), config)


class GFTTDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, config):
        cv_params = {}
        if 'nfeatures' in config:
            cv_params['maxCorners'] = config['nfeatures']
        if 'threshold' in config:
            cv_params['qualityLevel'] = float(config['threshold'])
        super().__init__(detector_name, logger, cv.GFTTDetector_create(**cv_params), config)


class MSERDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, config):
        cv_params = {}
        super().__init__(detector_name, logger, cv.MSER_create(**cv_params), config)


class AGASTDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, config):
        cv_params = {}
        if 'threshold' in config:
            cv_params['threshold'] = int(config['threshold'])
        super().__init__(detector_name, logger, cv.xfeatures2d.AgastFeatureDetector_create(**cv_params), config)


class BlobDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, config):
        super().__init__(detector_name, logger, cv.SimpleBlobDetector_create(), config)


class StarDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, config):
        cv_params = {}
        if 'threshold' in config:
            cv_params['responseThreshold'] = int(config['threshold'])
        super().__init__(detector_name, logger, cv.xfeatures2d.StarDetector_create(**cv_params), config)


class HarrisLaplaceDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, config):
        cv_params = {}
        if 'nfeatures' in config:
            cv_params['maxCorners'] = config['nfeatures']
        if 'threshold' in config:
            cv_params['corn_thresh'] = float(config['threshold'])
        super().__init__(detector_name, logger, cv.xfeatures2d.HarrisLaplaceFeatureDetector_create(**cv_params), config)


class MSDDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, config):
        cv_params = {}
        super().__init__(detector_name, logger, cv.xfeatures2d.MSDDetector_create(**cv_params), config)
