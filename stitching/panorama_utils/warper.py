from abc import ABC, abstractmethod
import cv2
import numpy as np
from stitching.panorama_utils.seam_finder import SeamFinder


class ImageWarper(ABC):
    _registry = {}

    def __init__(self, parameters, logger=None):
        self.parameters = parameters
        self.logger = logger

    @abstractmethod
    def warp_images(self, scaled_images, cameras, warped_image_scale,
                    seam_work_aspect, compensator):
        pass

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator

    @staticmethod
    def create(name, parameters, logger=None):
        warper_cls = ImageWarper._registry.get(name)
        if warper_cls is None:
            raise ValueError(f'Image warper {name} not implemented')
        return warper_cls(parameters, logger)


@ImageWarper.register('opencv')
class OpenCVImageWarper(ImageWarper):
    def __init__(self, parameters, logger=None):
        super().__init__(parameters, logger)
        self.warp_type = self.parameters['warp']
        self.seam_finder = SeamFinder.create(
            self.parameters['seam'], parameters, logger)

    def warp_images(self, scaled_images, cameras, warped_image_scale,
                    seam_work_aspect, compensator):
        corners = []
        masks_warped = []
        images_warped = []
        masks = []
        num_images = len(scaled_images)

        for i in range(num_images):
            um = cv2.UMat(255 * np.ones((scaled_images[i].shape[0], scaled_images[i].shape[1]), np.uint8))
            masks.append(um)

        warper = cv2.PyRotationWarper(self.warp_type, warped_image_scale * seam_work_aspect)
        K = None

        for idx in range(num_images):
            K = cameras[idx].K().astype(np.float32)
            K[0, 0] *= seam_work_aspect
            K[0, 2] *= seam_work_aspect
            K[1, 1] *= seam_work_aspect
            K[1, 2] *= seam_work_aspect
            corner, image_wp = warper.warp(scaled_images[idx], K, cameras[idx].R,
                                           cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
            corners.append(corner)
            images_warped.append(image_wp)
            p, mask_wp = warper.warp(masks[idx], K, cameras[idx].R,
                                     cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
            masks_warped.append(mask_wp.get())

        images_warped_f = [img.astype(np.float32) for img in images_warped]
        masks_warped = self.seam_finder.find(images_warped_f, corners, masks_warped)
        compensator.feed(corners, images_warped, masks_warped)

        return masks_warped, K


class ComposeWarper(ABC):
    _registry = {}

    def __init__(self, parameters, logger=None):
        self.parameters = parameters
        self.logger = logger

    @abstractmethod
    def setup(self, warped_image_scale):
        pass

    @abstractmethod
    def warp_roi(self, size, k, r):
        pass

    @abstractmethod
    def warp(self, img, k, r, interp_mode, border_mode):
        pass

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator

    @staticmethod
    def create(name, parameters, logger=None):
        warper_cls = ComposeWarper._registry.get(name)
        if warper_cls is None:
            raise ValueError(f'Compose warper {name} not implemented')
        if logger is not None:
            logger.info(f'Compose warper: {name} ({warper_cls.__name__})')
        return warper_cls(parameters, logger)


@ComposeWarper.register('opencv')
class OpenCVComposeWarper(ComposeWarper):
    def __init__(self, parameters, logger=None):
        super().__init__(parameters, logger)
        self.warper = None

    def setup(self, warped_image_scale):
        self.warper = cv2.PyRotationWarper(self.parameters['warp'], warped_image_scale)

    def warp_roi(self, size, k, r):
        return self.warper.warpRoi(size, k, r)

    def warp(self, img, k, r, interp_mode, border_mode):
        return self.warper.warp(img, k, r, interp_mode, border_mode)
