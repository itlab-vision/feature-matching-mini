from abc import ABC, abstractmethod
import cv2
import numpy as np
from stitching.panorama_utils.warper import ComposeWarper
from stitching.panorama_utils.blender import Blender


class PanoramaCreator(ABC):
    _registry = {}

    def __init__(self, parameters, logger=None):
        self.parameters = parameters
        self.logger = logger

    @abstractmethod
    def compose(self, images, cameras, masks_warped, warped_image_scale,
                work_scale, full_img_sizes, compensator):
        pass

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator

    @staticmethod
    def create(name, parameters, logger=None):
        creator_cls = PanoramaCreator._registry.get(name)
        if creator_cls is None:
            raise ValueError(f'Panorama creator {name} not implemented')
        return creator_cls(parameters, logger)


@PanoramaCreator.register('opencv')
class OpenCVPanoramaCreator(PanoramaCreator):
    def __init__(self, parameters, logger=None):
        super().__init__(parameters, logger)
        self.compose_megapix = self.parameters['compose_megapix']
        self.warper = ComposeWarper.create(
            self.parameters.get('compose_warper', 'opencv'), parameters, logger)
        self.blender = Blender.create(
            self.parameters['blend'], parameters, logger)

    def compose(self, images, cameras, masks_warped, warped_image_scale,
                work_scale, full_img_sizes, compensator):
        self.logger.info('START: Creating pano')
        compose_scale = 1
        is_compose_scale_set = False
        corners = []
        sizes = []
        for idx, full_image in enumerate(images):
            if not is_compose_scale_set:
                if self.compose_megapix > 0:
                    compose_scale = min(1.0, np.sqrt(self.compose_megapix * 1e6
                                                     / (full_image.shape[0] * full_image.shape[1])))
                is_compose_scale_set = True
                compose_work_aspect = compose_scale / work_scale
                warped_image_scale *= compose_work_aspect
                self.warper.setup(warped_image_scale)
                for i in range(0, len(images)):
                    cameras[i].focal *= compose_work_aspect
                    cameras[i].ppx *= compose_work_aspect
                    cameras[i].ppy *= compose_work_aspect
                    sz = (int(round(full_img_sizes[i][0] * compose_scale)),
                          int(round(full_img_sizes[i][1] * compose_scale)))
                    K = cameras[i].K().astype(np.float32)
                    roi = self.warper.warp_roi(sz, K, cameras[i].R)
                    corners.append(roi[0:2])
                    sizes.append(roi[2:4])
            if abs(compose_scale - 1) > 1e-1:
                img = cv2.resize(src=full_image, dsize=None,
                                 fx=compose_scale, fy=compose_scale,
                                 interpolation=cv2.INTER_LINEAR_EXACT)
            else:
                img = full_image
            K = cameras[idx].K().astype(np.float32)
            corner, image_warped = self.warper.warp(img, K, cameras[idx].R,
                                                    cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
            mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
            p, mask_warped = self.warper.warp(mask, K, cameras[idx].R,
                                              cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
            compensator.apply(idx, corners[idx], image_warped, mask_warped)
            image_warped_s = image_warped.astype(np.int16)
            dilated_mask = cv2.dilate(masks_warped[idx], None)
            seam_mask = cv2.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]),
                                   0, 0, cv2.INTER_LINEAR_EXACT)
            mask_warped = cv2.bitwise_and(seam_mask, mask_warped)
            if not self.blender.prepared:
                dst_sz = cv2.detail.resultRoi(corners=corners, sizes=sizes)
                self.blender.prepare(dst_sz)
            self.blender.feed(cv2.UMat(image_warped_s), mask_warped, corners[idx])
        result, result_mask = self.blender.blend()
        self.logger.info('FINISH: Creating pano')
        return result
