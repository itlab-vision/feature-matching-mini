import cv2 as cv
from abc import ABC, abstractmethod


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
            if key:
                Detector._METHODS[key] = cls

    @staticmethod
    def create(detector_name, logger, config=None):
        if config is None:
            config = {}

        if detector_name not in Detector._METHODS:
            raise ValueError(f"Detector '{detector_name}' not found."
                             f" Available: {list(Detector._METHODS.keys())}")

        return Detector._METHODS[detector_name](detector_name, logger, config)

    @abstractmethod
    def detect(self, img):
        pass


class OpenCVDetector(Detector, register=False):
    def __init__(self, detector_name, logger, extractor, config):
        super().__init__(logger, detector_name)
        self._extractor = extractor

    def detect(self, img):
        if img is None:
            self._logger.error("Input image is None. Detection aborted.")
            return {'kp': ()}

        self._logger.info(f"Detecting keypoints with {self._detector_name}")
        kp = self._extractor.detect(img, None)

        if kp:
            self._logger.info(f"{self._detector_name} found {len(kp)} points")
        else:
            self._logger.warning(f"{self._detector_name} found 0 points")
        return {'kp': kp}


class SIFTDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, config):
        super().__init__(detector_name, logger, cv.SIFT_create(**config), config)


class ORBDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, config):
        super().__init__(detector_name, logger, cv.ORB_create(**config), config)


class FASTDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, config):
        super().__init__(detector_name, logger, cv.FastFeatureDetector_create(**config), config)


class AKAZEDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, config):
        super().__init__(detector_name, logger, cv.xfeatures2d.AKAZE_create(**config), config)


class BRISKDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, config):
        super().__init__(detector_name, logger, cv.xfeatures2d.BRISK_create(**config), config)


class KAZEDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, config):
        super().__init__(detector_name, logger, cv.xfeatures2d.KAZE_create(**config), config)


class GFTTDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, config):
        super().__init__(detector_name, logger, cv.GFTTDetector_create(**config), config)


class MSERDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, config):
        super().__init__(detector_name, logger, cv.MSER_create(**config), config)


class AGASTDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, config):
        super().__init__(detector_name, logger, cv.xfeatures2d.AgastFeatureDetector_create(**config), config)


class BlobDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, config):
        super().__init__(detector_name, logger, cv.SimpleBlobDetector_create(**config), config)


class StarDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, config):
        super().__init__(detector_name, logger, cv.xfeatures2d.StarDetector_create(**config), config)


class HarrisLaplaceDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, config):
        super().__init__(detector_name, logger, cv.xfeatures2d.HarrisLaplaceFeatureDetector_create(**config), config)


class MSDDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, config):
        super().__init__(detector_name, logger, cv.xfeatures2d.MSDDetector_create(**config), config)
