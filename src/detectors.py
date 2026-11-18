from abc import ABC, abstractmethod

from src.algorithms import DNN_DESCRIPTORS, DNN_DETECTORS, DNN_PIPELINES


class Detector(ABC):
    _METHODS = {}

    def __init__(self, logger, detector_name='sift'):
        self._detector_name = detector_name
        self._logger = logger

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls._METHODS[name.lower()] = subclass
            return subclass

        return decorator

    @staticmethod
    def create(detector_name, logger, config=None):
        if detector_name in DNN_PIPELINES:
            return Detector._METHODS["dnnpipeline"](detector_name, logger, config)

        if detector_name in DNN_DETECTORS and detector_name in DNN_DESCRIPTORS:
            return Detector._METHODS["dnnfeatureextractors"](detector_name, logger, config)

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
