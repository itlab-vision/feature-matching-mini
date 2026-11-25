from abc import ABC, abstractmethod
import cv2

SEAM_FIND_DEFAULT_CHOICES = {
    'voronoi': cv2.detail.SeamFinder_VORONOI_SEAM,
    'no': cv2.detail.SeamFinder_NO,
}


class SeamFinder(ABC):
    _registry = {}

    def __init__(self, parameters, logger=None):
        self.parameters = parameters
        self.logger = logger

    @abstractmethod
    def find(self, images_warped_f, corners, masks_warped):
        pass

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator

    @staticmethod
    def create(name, parameters, logger=None):
        seam_finder_cls = SeamFinder._registry.get(name)
        if seam_finder_cls is None:
            raise ValueError(f'Seam finder {name} not implemented')
        return seam_finder_cls(parameters, logger)


class OpenCVSeamFinder(SeamFinder):
    def __init__(self, parameters, seam_finder, logger=None):
        super().__init__(parameters, logger)
        self.seam_finder = seam_finder

    def find(self, images_warped_f, corners, masks_warped):
        return self.seam_finder.find(images_warped_f, corners, masks_warped)


@SeamFinder.register('gc_color')
class GraphCutColorSeamFinder(OpenCVSeamFinder):
    def __init__(self, parameters, logger=None):
        seam_finder = cv2.detail_GraphCutSeamFinder('COST_COLOR')
        super().__init__(parameters, seam_finder, logger)


@SeamFinder.register('gc_colorgrad')
class GraphCutColorGradSeamFinder(OpenCVSeamFinder):
    def __init__(self, parameters, logger=None):
        seam_finder = cv2.detail_GraphCutSeamFinder('COST_COLOR_GRAD')
        super().__init__(parameters, seam_finder, logger)


@SeamFinder.register('dp_color')
class DpColorSeamFinder(OpenCVSeamFinder):
    def __init__(self, parameters, logger=None):
        seam_finder = cv2.detail_DpSeamFinder('COLOR')
        super().__init__(parameters, seam_finder, logger)


@SeamFinder.register('dp_colorgrad')
class DpColorGradSeamFinder(OpenCVSeamFinder):
    def __init__(self, parameters, logger=None):
        seam_finder = cv2.detail_DpSeamFinder('COLOR_GRAD')
        super().__init__(parameters, seam_finder, logger)


class DefaultSeamFinder(OpenCVSeamFinder):
    def __init__(self, parameters, logger=None):
        seam_finder = cv2.detail.SeamFinder_createDefault(
            SEAM_FIND_DEFAULT_CHOICES[parameters['seam']])
        super().__init__(parameters, seam_finder, logger)


SeamFinder.register('voronoi')(DefaultSeamFinder)
SeamFinder.register('no')(DefaultSeamFinder)
