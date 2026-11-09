from abc import ABC, abstractmethod

from src.algorithms import ALL_DETECTORS, DNN_DESCRIPTORS, DNN_DETECTORS, DNN_PIPELINES


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
