from abc import ABC, abstractmethod
import cv2
import numpy as np


class ImageTransformer(ABC):
    _registry = {}

    def __init__(self, parameters, logger=None):
        self.parameters = parameters
        self.logger = logger

    @abstractmethod
    def transform(self, images):
        pass

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator

    @staticmethod
    def create(name, parameters, logger=None):
        transformer_cls = ImageTransformer._registry.get(name)
        if transformer_cls is None:
            raise ValueError(f'Image transformer {name} not implemented')
        return transformer_cls(parameters, logger)


@ImageTransformer.register('opencv')
class OpenCVImageTransformer(ImageTransformer):
    def __init__(self, parameters, logger=None):
        super().__init__(parameters, logger)
        self.work_megapix = self.parameters['work_megapix']
        self.seam_megapix = self.parameters['seam_megapix']
        self.work_scale = None
        self.seam_work_aspect = None
        self.full_img_sizes = []

    def transform(self, images):
        transform_images_for_features = []
        transform_images_seam_scaling = []
        self.full_img_sizes = []
        is_work_scale_set = False
        is_seam_scale_set = False
        seam_scale = None

        for full_image in images:
            self.full_img_sizes.append((full_image.shape[1], full_image.shape[0]))

            if self.work_megapix < 0:
                img = full_image
                self.work_scale = 1
                is_work_scale_set = True
            else:
                if is_work_scale_set is False:
                    self.work_scale = min(1.0, np.sqrt(self.work_megapix * 1e6
                                                       / (full_image.shape[0] * full_image.shape[1])))
                    is_work_scale_set = True
                img = cv2.resize(src=full_image, dsize=None,
                                 fx=self.work_scale, fy=self.work_scale,
                                 interpolation=cv2.INTER_LINEAR_EXACT)

            if is_seam_scale_set is False:
                if self.seam_megapix > 0:
                    seam_scale = min(1.0, np.sqrt(self.seam_megapix * 1e6
                                                  / (full_image.shape[0] * full_image.shape[1])))
                else:
                    seam_scale = 1.0
                self.seam_work_aspect = seam_scale / self.work_scale
                is_seam_scale_set = True

            transform_images_for_features.append(img)
            img_seam = cv2.resize(src=full_image, dsize=None,
                                  fx=seam_scale, fy=seam_scale,
                                  interpolation=cv2.INTER_LINEAR_EXACT)
            transform_images_seam_scaling.append(img_seam)

        return transform_images_for_features, transform_images_seam_scaling
