from abc import abstractmethod

from src.dnn_extractors import DNNFeatureExtractors
from src.dnn_matchers import DNNMatcher


class DNNPipeline(DNNFeatureExtractors, DNNMatcher, register=False):
    def __init__(self, extractor_name, logger, config=None):
        if config is None:
            config = {}

        DNNFeatureExtractors.__init__(self, extractor_name, logger, config)
        DNNMatcher.__init__(self, extractor_name, logger, config, extractor_name)

    def detect(self, img):
        return {'image': img}

    def compute(self, img, features=None):
        return features

    @abstractmethod
    def match(self, features0, features1):
        pass
