from abc import ABC, abstractmethod

from src.algorithms import DNN_PIPELINES, DNN_DESCRIPTORS, DNN_DETECTORS


class Descriptor(ABC):
    _METHODS = {}

    def __init__(self, logger, descriptor_name='sift'):
        self._descriptor_name = descriptor_name
        self._logger = logger

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls._METHODS[name.lower()] = subclass
            return subclass

        return decorator

    @staticmethod
    def create(descriptor_name, logger, config=None):
        if config is None:
            config = {}

        descriptor_name = descriptor_name.lower()

        if descriptor_name in DNN_PIPELINES:
            return Descriptor._METHODS["dnnpipeline"](descriptor_name, logger, config)

        if descriptor_name in DNN_DESCRIPTORS and descriptor_name not in DNN_DETECTORS:
            return Descriptor._METHODS['dnn'](descriptor_name, logger, config)

        if descriptor_name in DNN_DESCRIPTORS and descriptor_name in DNN_DETECTORS:
            return Descriptor._METHODS["dnnfeatureextractors"](descriptor_name, logger, config)

        descriptor_class_name = descriptor_name.lower()
        if descriptor_class_name not in Descriptor._METHODS:
            raise ValueError(f"Descriptor '{descriptor_class_name}' not found."
                             f" Available: {list(Descriptor._METHODS.keys())}")

        return Descriptor._METHODS[descriptor_class_name](descriptor_class_name, logger, config)

    @property
    @abstractmethod
    def default_norm(self):
        pass

    @abstractmethod
    def compute(self, img, kp):
        pass
