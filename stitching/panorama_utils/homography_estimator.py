from abc import ABC, abstractmethod
import cv2
import numpy as np
from stitching.opencv_constants_auxiliary import BA_COST_CHOICES, ESTIMATOR_CHOICES


class HomographyEstimator(ABC):
    _registry = {}

    def __init__(self, parameters, logger=None):
        self.parameters = parameters
        self.logger = logger

    @abstractmethod
    def estimate(self, features, pairs):
        pass

    @abstractmethod
    def refine(self, features, pairs, cameras):
        pass

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator

    @staticmethod
    def create(name, parameters, logger=None):
        estimator_cls = HomographyEstimator._registry.get(name)
        if estimator_cls is None:
            raise ValueError(f'Homography estimator {name} not implemented')
        return estimator_cls(parameters, logger)


class OpenCVHomographyEstimator(HomographyEstimator):
    def __init__(self, parameters, logger=None):
        super().__init__(parameters, logger)
        self.estimator = self._create_estimator()
        self.adjuster = BA_COST_CHOICES[self.parameters['ba']]()
        self.conf_thresh = self.parameters['conf_thresh']
        self.ba_refine_mask = self.parameters['ba_refine_mask']

    def _create_estimator(self):
        return ESTIMATOR_CHOICES[self.parameters['estimator']]()

    def estimate(self, features, pairs):
        status, cameras = self.estimator.apply(features, pairs, None)
        if not status:
            raise RuntimeError('Homography estimation failed')
        for cam in cameras:
            cam.R = cam.R.astype(np.float32)
        return cameras

    def refine(self, features, pairs, cameras):
        self.adjuster.setConfThresh(self.conf_thresh)
        refine_mask = np.zeros((3, 3), np.uint8)
        if self.ba_refine_mask[0] == 'x':
            refine_mask[0, 0] = 1
        if self.ba_refine_mask[1] == 'x':
            refine_mask[0, 1] = 1
        if self.ba_refine_mask[2] == 'x':
            refine_mask[0, 2] = 1
        if self.ba_refine_mask[3] == 'x':
            refine_mask[1, 1] = 1
        if self.ba_refine_mask[4] == 'x':
            refine_mask[1, 2] = 1
        self.adjuster.setRefinementMask(refine_mask)
        adjusted = [self._copy_camera(cam) for cam in cameras]
        try:
            status, refined = self.adjuster.apply(features, pairs, adjusted)
        except cv2.error as error:
            if self.logger is not None:
                self.logger.warning(f'Bundle adjuster failed, cameras left unrefined: {error}')
            return cameras
        if not status:
            raise RuntimeError('Camera parameters adjusting failed')
        return refined

    @staticmethod
    def _copy_camera(cam):
        copy = cv2.detail.CameraParams()
        copy.focal = cam.focal
        copy.aspect = cam.aspect
        copy.ppx = cam.ppx
        copy.ppy = cam.ppy
        copy.R = cam.R.copy()
        copy.t = cam.t.copy()
        return copy


@HomographyEstimator.register('homography')
class HomographyBasedEstimator(OpenCVHomographyEstimator):
    pass


@HomographyEstimator.register('affine')
class AffineBasedEstimator(OpenCVHomographyEstimator):
    pass
