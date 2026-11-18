from abc import ABC, abstractmethod
import numpy as np

from src.algorithms import ALL_DESCRIPTORS, ALL_DETECTORS, DNN_PIPELINES

from stitching.panorama_utils.utils import (
    import_src,
    resolve_src_names,
    src_stage_config,
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
            name = method_name.lower().removesuffix('_opencv')
            if name not in ALL_DETECTORS and name not in ALL_DESCRIPTORS:
                raise ValueError(f'Method {method_name} not implemented')
            extractor_cls = PipelineFeatureExtractor
            parameters = {**parameters, 'features': method_name}
        if logger is not None:
            logger.info(f'Feature extractor: {method_name} ({extractor_cls.__name__})')
        return extractor_cls(parameters, logger)


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
        self.descriptor = None
        self.is_pipeline = self.detector_name in DNN_PIPELINES
        if not self.is_pipeline:
            detector_cfg = src_stage_config(parameters, 'detector', self.detector_name)
            descriptor_cfg = src_stage_config(parameters, 'descriptor', self.descriptor_name)
            detector_backend = detector_cfg['backend']
            descriptor_backend = descriptor_cfg['backend']
            validate_detector_descriptor(self.detector_name, self.descriptor_name,
                                         detector_backend, descriptor_backend)
            Detector, Descriptor, _ = import_src(detector_backend, descriptor_backend)
            self.detector = Detector.create(self.detector_name, logger, detector_cfg)
            self.descriptor = Descriptor.create(self.descriptor_name, logger, descriptor_cfg)

    def extract_src(self, image):
        if self.is_pipeline:
            return {'keypoints': (), 'descriptors': ()}
        features = self.detector.detect(image)
        features = self.descriptor.compute(image, features)
        height, width = image.shape[:2]
        features.setdefault('extractor_name', self.descriptor_name)
        features.setdefault('height', height)
        features.setdefault('width', width)
        features.setdefault('image_size', np.array([[width, height]]))
        return features

    def extract(self, image):
        return src_features_to_image_features(self.extract_src(image), image)
