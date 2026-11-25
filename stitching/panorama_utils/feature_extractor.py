from abc import ABC, abstractmethod
import cv2
import numpy as np
import importlib

from stitching.panorama_utils.utils import (
    SRC_EXTRACTOR_NAMES,
    SRC_PIPELINES,
    import_src,
    resolve_src_names,
    src_config,
    src_features_to_image_features,
    validate_detector_descriptor,
)


class FeatureExtractor(ABC):
    _registry = {}

    def __init__(self, parameters, logger):
        self.parametrs = parameters
        self.logger = logger

    @abstractmethod
    def extract(self, image):
        pass

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator

    @staticmethod
    def create(method_name, parameters, logger):
        extractor_cls = FeatureExtractor._registry.get(method_name)
        if extractor_cls is None:
            raise ValueError(f'Method {method_name} not implemented')
        if logger is not None:
            logger.info(f'Feature extractor: {method_name} ({extractor_cls.__name__})')
        return extractor_cls(parameters, logger)


class OpenCVFeatureExtractor(FeatureExtractor):
    def __init__(self, parameters, logger):
        super().__init__(parameters, logger)
        self.feature = None

    def extract(self, image):
        return cv2.detail.computeImageFeatures2(self.feature, image)


@FeatureExtractor.register('orb')
class ORBFeatureExtractor(OpenCVFeatureExtractor):
    def __init__(self, parameters, logger):
        super().__init__(parameters, logger)
        self.feature = cv2.ORB.create()


@FeatureExtractor.register('sift')
class SIFTFeatureExtractor(OpenCVFeatureExtractor):
    def __init__(self, parameters, logger):
        super().__init__(parameters, logger)
        self.feature = cv2.SIFT_create()


@FeatureExtractor.register('brisk')
class BRISKFeatureExtractor(OpenCVFeatureExtractor):
    def __init__(self, parameters, logger):
        super().__init__(parameters, logger)
        self.feature = cv2.BRISK_create()


@FeatureExtractor.register('akaze')
class AKAZEFeatureExtractor(OpenCVFeatureExtractor):
    def __init__(self, parameters, logger):
        super().__init__(parameters, logger)
        self.feature = cv2.AKAZE_create()


@FeatureExtractor.register('superpoint')
class SuperPointFeatureExtractor(FeatureExtractor):
    def __init__(self, parameters, logger):
        super().__init__(parameters, logger)
        self.torch = importlib.import_module('torch')
        self.device = self.torch.device('cuda' if self.torch.cuda.is_available() else 'cpu')
        self.model = importlib.import_module('lightglue').SuperPoint(max_num_keypoints=None).eval().to(self.device)
        self.score_thresh = parameters.get('score_thresh', 0.15)

    def numpy_image_to_torch(self, image):
        if image.ndim == 3:
            image = image.transpose((2, 0, 1))
        elif image.ndim == 2:
            image = image[None]
        else:
            raise ValueError(f"Not an image: {image.shape}")
        return self.torch.tensor(image / 255.0, dtype=self.torch.float)

    def extract(self, image):
        feats = self.model.extract(self.numpy_image_to_torch(image).to(self.device))
        keypoints, descriptors, scores = (feats['keypoints'].cpu().numpy()[0],
                                          feats['descriptors'].detach().cpu().float(),
                                          feats['keypoint_scores'].cpu().numpy()[0])
        descriptors = descriptors.contiguous()
        numpy_array = descriptors.numpy().copy()
        numpy_array = np.ascontiguousarray(numpy_array)
        mask = scores >= self.score_thresh
        keypoints = keypoints[mask]
        scores = scores[mask]
        numpy_array = numpy_array[:, mask, :]
        features_cv = cv2.detail.ImageFeatures()
        cv_keypoints = []
        for i, (score, keypoint) in enumerate(zip(scores, keypoints)):
            cv_keypoints.append(cv2.KeyPoint(int(keypoint[0]), int(keypoint[1]), 1.0, -1, score, 0, i))
        features_cv.keypoints = tuple(cv_keypoints)
        features_cv.descriptors = cv2.Mat(numpy_array)
        features_cv.descriptors = np.squeeze(features_cv.descriptors.get())
        features_cv.img_idx = 80
        features_cv.img_size = (image.shape[1], image.shape[0])
        return features_cv


@FeatureExtractor.register('pipeline')
class PipelineFeatureExtractor(FeatureExtractor):
    """Feature extractor backed by the src feature-matching library.

    Usage in config:
        detector: 'xfeat'                 # src detector name
        descriptor: 'xfeat'               # src descriptor name
        src_config:
          detector: {nfeatures: 2000}     # optional detector params
          descriptor: {}                  # optional descriptor params

    Methods that bundle a detector+descriptor pair (e.g. xfeat,
    disk_lightglue) are given for both detector and descriptor.
    Legacy keys `features` / `src_detector` / `src_descriptor` are still
    supported (e.g. features: 'xfeat', features: 'fast', features: 'hardnet').
    """

    def __init__(self, parameters, logger):
        super().__init__(parameters, logger)
        self.detector_name, self.descriptor_name = resolve_src_names(parameters)
        self.cfg = src_config(parameters)
        self.is_pipeline = self.detector_name in SRC_PIPELINES
        if not self.is_pipeline:
            Detector, Descriptor, _ = import_src()
            validate_detector_descriptor(self.detector_name, self.descriptor_name)
            detector_cfg = dict(self.cfg.get('detector', {}) or {})
            descriptor_cfg = dict(self.cfg.get('descriptor', {}) or {})
            self.logger.info(
                f'Src feature extractor: detector={self.detector_name} descriptor={self.descriptor_name}')
            self.detector = Detector.create(self.detector_name, self.logger, detector_cfg)
            self.descriptor = Descriptor.create(self.descriptor_name, self.logger, descriptor_cfg)

    def extract(self, image):
        if self.is_pipeline:
            return src_features_to_image_features({'keypoints': (), 'descriptors': ()},
                                                  image, 80)
        features = self.detector.detect(image)
        features = self.descriptor.compute(image, features)
        return src_features_to_image_features(features, image, 80)


for _src_extractor_name in SRC_EXTRACTOR_NAMES:
    FeatureExtractor.register(_src_extractor_name)(PipelineFeatureExtractor)
