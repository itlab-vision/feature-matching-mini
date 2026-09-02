import cv2 as cv
from abc import ABC, abstractmethod

from src.algorithms import ALL_DESCRIPTORS


class Descriptor(ABC):
    _METHODS = {}

    def __init__(self, logger, descriptor_name='sift'):
        self._descriptor_name = descriptor_name
        self._logger = logger

    def __init_subclass__(cls, register=True, **kwargs):
        super().__init_subclass__(**kwargs)

        if hasattr(cls, '_EXTRACTOR_CLASSES'):
            for key in cls._EXTRACTOR_CLASSES.keys():
                Descriptor._METHODS[key] = cls

        elif register:
            key = cls.__name__.replace("Descriptor", "").lower()

            for key_ in ALL_DESCRIPTORS:
                if key_.replace("_", "") == key:
                    key = key_
                    break
            if key:
                Descriptor._METHODS[key] = cls

    @staticmethod
    def create(descriptor_name, logger, config=None):
        if config is None:
            config = {}

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


class OpenCVDescriptor(Descriptor, register=False):
    def __init__(self, descriptor_name, logger, extractor, config):
        super().__init__(logger, descriptor_name)
        self._extractor = extractor
        self._nfeatures = config.get('nfeatures')

    @property
    def default_norm(self):
        return self._extractor.defaultNorm()

    def compute(self, img, features):
        if img is None:
            self._logger.error("Input image is None. Detection aborted.")
            return {'kp': (), 'des': ()}

        self._logger.info(f"Computing {self._descriptor_name} descriptors")
        kp, des = self._extractor.compute(img, features.get('kp'))

        if des is not None and self._nfeatures is not None and len(des) > self._nfeatures:
            kp = kp[:self._nfeatures]
            des = des[:self._nfeatures]

        if des is not None:
            self._logger.info(f"{self._descriptor_name} computed {len(des)} descriptors")
        else:
            self._logger.warning(f"{self._descriptor_name} computed 0 descriptors")
        return {'kp': kp, 'des': des}


class SIFTDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name, logger, config):
        cv_params = {}
        if 'nfeatures' in config:
            cv_params['nfeatures'] = config['nfeatures']
        super().__init__(descriptor_name, logger, cv.SIFT_create(**cv_params), config)


class ORBDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name, logger, config):
        cv_params = {}
        if 'nfeatures' in config:
            cv_params['nfeatures'] = config['nfeatures']
        super().__init__(descriptor_name, logger, cv.ORB_create(**cv_params), config)


class AKAZEDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name, logger, config):
        super().__init__(descriptor_name, logger, cv.xfeatures2d.AKAZE_create(**config), config)


class BRISKDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name, logger, config):
        super().__init__(descriptor_name, logger, cv.xfeatures2d.BRISK_create(**config), config)


class KAZEDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name, logger, config):
        super().__init__(descriptor_name, logger, cv.xfeatures2d.KAZE_create(**config), config)


class BRIEFDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name, logger, config):
        cv_params = {}
        if 'bytes' in config:
            cv_params['bytes'] = config['bytes']
        super().__init__(descriptor_name, logger, cv.xfeatures2d.BriefDescriptorExtractor_create(**cv_params), config)


class FREAKDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name, logger, config):
        super().__init__(descriptor_name, logger, cv.xfeatures2d.FREAK_create(**config), config)


class DAISYDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name, logger, config):
        super().__init__(descriptor_name, logger, cv.xfeatures2d.DAISY_create(**config), config)


class LATCHDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name, logger, config):
        cv_params = {}
        if 'bytes' in config:
            cv_params['bytes'] = config['bytes']
        super().__init__(descriptor_name, logger, cv.xfeatures2d.LATCH_create(**cv_params), config)


class BEBLIDDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name, logger, config):
        scale_factor = config.get('scale_factor', 0.75)
        cv_params = {'scale_factor': scale_factor}
        if 'n_bits' in config:
            cv_params['n_bits'] = config['n_bits']
        super().__init__(descriptor_name, logger, cv.xfeatures2d.BEBLID_create(**cv_params), config)


class TEBLIDDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name, logger, config):
        scale_factor = config.get('scale_factor', 0.75)
        cv_params = {'scale_factor': scale_factor}
        if 'n_bits' in config:
            cv_params['n_bits'] = config['n_bits']
        super().__init__(descriptor_name, logger, cv.xfeatures2d.TEBLID_create(**cv_params), config)


class VGGDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name, logger, config):
        super().__init__(descriptor_name, logger, cv.xfeatures2d.VGG_create(**config), config)


class BoostDescDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name, logger, config):
        super().__init__(descriptor_name, logger, cv.xfeatures2d.BoostDesc_create(**config), config)
