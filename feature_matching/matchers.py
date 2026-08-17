from abc import ABC, abstractmethod
import cv2 as cv


class Matcher(ABC):
    _METHODS = {}

    def __init__(self, logger, matcher_name, descriptor_name):
        self.matcher_name = matcher_name
        self.descriptor_name = descriptor_name
        self.logger = logger

    def __init_subclass__(cls, register=True, **kwargs):
        super().__init_subclass__(**kwargs)

        if register:
            key = cls.__name__.replace("Matcher", "").lower()
            if key:
                Matcher._METHODS[key] = cls

    @staticmethod
    def create(matcher_name, logger, descriptor_name, config):
        matcher_class_name = Matcher._METHODS.get(matcher_name.lower())
        if not matcher_class_name:
            raise ValueError(f"Matcher '{matcher_name}' not found."
                             f" Available: {list(Matcher._METHODS.keys())}")

        return Matcher._METHODS[matcher_name](logger, matcher_name, descriptor_name, config)

    @abstractmethod
    def match(self, features1, features2):
        pass

    @abstractmethod
    def _init_matcher(self):
        pass


class OpenCVMatcher(Matcher, register=False):
    _MODE = {'simple', 'knn'}

    def __init__(self, logger, matcher_name, descriptor_name, config):
        super().__init__(logger, matcher_name, descriptor_name)
        self.mode = config.get('mode', 'simple')
        self.k = config.get('k', 1)

    def match(self, features1, features2):
        des1 = features1.get('des')
        des2 = features2.get('des')

        if des1 is None or des2 is None:
            return {'matches': ()}

        matcher = self._init_matcher()
        if self.mode == 'simple':
            return {'matches': matcher.match(des1, des2)}
        elif self.mode == 'knn':
            return {'matches': matcher.knnMatch(des1, des2, self.k)}
        else:
            raise ValueError(f"Mode '{self.mode}' is not supported.")


class BFMatcher(OpenCVMatcher):
    def __init__(self, logger, matcher_name, descriptor_name, config):
        super().__init__(logger, matcher_name, descriptor_name, config)

    def _init_matcher(self):
        return cv.BFMatcher(self.descriptor_name.default_norm)


class FLANNMatcher(OpenCVMatcher):
    def __init__(self, logger, matcher_name, descriptor_name, config):
        super().__init__(logger, matcher_name, descriptor_name, config)
        self.index_params = config.get('index_params')
        self.search_params = config.get('search_params')

        if self.index_params is None:
            self.index_params = self._get_default_index_params()
        elif not isinstance(self.index_params, dict):
            raise TypeError("index_params must be dict")
        if self.search_params is None:
            self.search_params = {'checks': 50}
        elif not isinstance(self.search_params, dict):
            raise TypeError("search_params must be dict")

    def _get_default_index_params(self):
        if self.descriptor_name.default_norm == cv.NORM_HAMMING:
            return {
                'algorithm': 6,
                'table_number': 6,
                'key_size': 12,
                'multi_probe_level': 1
            }
        else:
            return {
                'algorithm': 1,
                'trees': 5
            }

    def _init_matcher(self):
        return cv.FlannBasedMatcher(self.index_params, self.search_params)
