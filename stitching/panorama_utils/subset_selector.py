from abc import ABC, abstractmethod
import cv2


class SubsetSelector(ABC):
    _registry = {}

    def __init__(self, parameters, logger=None):
        self.parameters = parameters
        self.logger = logger

    @abstractmethod
    def select(self, features, pairs, scaled_imgs, full_images, full_img_sizes):
        pass

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator

    @staticmethod
    def create(name, parameters, logger=None):
        selector_cls = SubsetSelector._registry.get(name)
        if selector_cls is None:
            raise ValueError(f'Subset selector {name} not implemented')
        return selector_cls(parameters, logger)


@SubsetSelector.register('opencv')
class OpenCVSubsetSelector(SubsetSelector):
    def __init__(self, parameters, logger=None):
        super().__init__(parameters, logger)
        self.conf_thresh = self.parameters['conf_thresh']

    def select(self, features, pairs, scaled_imgs, full_images, full_img_sizes):
        indices = cv2.detail.leaveBiggestComponent(features, pairs, self.conf_thresh)
        img_subset = [scaled_imgs[i] for i in indices]
        full_img_sizes_subset = [full_img_sizes[i] for i in indices]
        full_images_subset = [full_images[i] for i in indices]
        return img_subset, full_images_subset, full_img_sizes_subset
