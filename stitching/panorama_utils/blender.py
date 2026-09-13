from abc import ABC, abstractmethod
import cv2
import numpy as np


class Blender(ABC):
    _registry = {}

    def __init__(self, parameters, logger=None):
        self.parameters = parameters
        self.logger = logger
        self.blender = None
        self.prepared = False

    def prepare(self, dst_sz):
        self._create_blender(dst_sz)
        self.blender.prepare(dst_sz)
        self.prepared = True

    @abstractmethod
    def _create_blender(self, dst_sz):
        pass

    def feed(self, img, mask, corner):
        self.blender.feed(img, mask, corner)

    def blend(self, result=None, result_mask=None):
        return self.blender.blend(result, result_mask)

    def _compute_blend_width(self, dst_sz):
        return np.sqrt(dst_sz[2] * dst_sz[3]) * self.parameters['blend_strength'] / 100

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator

    @staticmethod
    def create(name, parameters, logger=None):
        blender_cls = Blender._registry.get(name)
        if blender_cls is None:
            raise ValueError(f'Blender {name} not implemented')
        if logger is not None:
            logger.info(f'Blender: {name} ({blender_cls.__name__})')
        return blender_cls(parameters, logger)


@Blender.register('multiband')
class MultiBandBlender(Blender):
    def _create_blender(self, dst_sz):
        blend_width = self._compute_blend_width(dst_sz)
        if blend_width < 1:
            self.blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)
            return
        self.blender = cv2.detail_MultiBandBlender()
        num_bands = (np.log(blend_width) / np.log(2.) - 1.).astype(np.int32)
        self.blender.setNumBands(num_bands)


@Blender.register('feather')
class FeatherBlender(Blender):
    def _create_blender(self, dst_sz):
        blend_width = self._compute_blend_width(dst_sz)
        if blend_width < 1:
            self.blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)
            return
        self.blender = cv2.detail_FeatherBlender()
        self.blender.setSharpness(1. / blend_width)


@Blender.register('no')
class NoBlender(Blender):
    def _create_blender(self, dst_sz):
        self.blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)
