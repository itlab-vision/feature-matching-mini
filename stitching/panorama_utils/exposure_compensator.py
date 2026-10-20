from abc import ABC, abstractmethod
import cv2
from stitching.opencv_constants_auxiliary import EXPOS_COMP_CHOICES


class ExposureCompensator(ABC):
    _registry = {}

    def __init__(self, parameters, logger=None):
        self.parameters = parameters
        self.logger = logger

    @abstractmethod
    def feed(self, corners, images, masks):
        pass

    @abstractmethod
    def apply(self, img_idx, corner, image_warped, mask_warped):
        pass

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator

    @staticmethod
    def create(name, parameters, logger=None):
        compensator_cls = ExposureCompensator._registry.get(name)
        if compensator_cls is None:
            raise ValueError(f'Exposure compensator {name} not implemented')
        return compensator_cls(parameters, logger)


class OpenCVExposureCompensator(ExposureCompensator):
    def __init__(self, parameters, compensator, logger=None):
        super().__init__(parameters, logger)
        self.compensator = compensator

    def feed(self, corners, images, masks):
        self.compensator.feed(corners, images, masks)

    def apply(self, img_idx, corner, image_warped, mask_warped):
        self.compensator.apply(img_idx, corner, image_warped, mask_warped)


@ExposureCompensator.register('channel')
class ChannelsCompensator(OpenCVExposureCompensator):
    def __init__(self, parameters, logger=None):
        compensator = cv2.detail_ChannelsCompensator(parameters['expos_comp_nr_feeds'])
        super().__init__(parameters, compensator, logger)


@ExposureCompensator.register('channel_blocks')
class BlocksChannelsCompensator(OpenCVExposureCompensator):
    def __init__(self, parameters, logger=None):
        compensator = cv2.detail_BlocksChannelsCompensator(
            parameters['expos_comp_block_size'],
            parameters['expos_comp_block_size'],
            parameters['expos_comp_nr_feeds']
        )
        super().__init__(parameters, compensator, logger)


class DefaultCompensator(OpenCVExposureCompensator):
    def __init__(self, parameters, logger=None):
        compensator = cv2.detail.ExposureCompensator_createDefault(
            EXPOS_COMP_CHOICES[parameters['expos_comp']]
        )
        super().__init__(parameters, compensator, logger)


ExposureCompensator.register('gain')(DefaultCompensator)
ExposureCompensator.register('gain_blocks')(DefaultCompensator)
ExposureCompensator.register('no')(DefaultCompensator)
