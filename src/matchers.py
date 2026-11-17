from abc import ABC, abstractmethod

from src.algorithms import ALL_MATCHERS, DNN_PIPELINES, DNN_MATCHERS, get_default_backend


class Matcher(ABC):
    _METHODS = {}

    def __init__(self, matcher_name, logger, config=None, descriptor_class=None):
        if config is None:
            config = {}

        self._matcher_name = matcher_name
        self._descriptor_class = descriptor_class
        self._logger = logger

    def __init_subclass__(cls, register=True, **kwargs):
        super().__init_subclass__(**kwargs)

        if hasattr(cls, '_MATCHER_CLASSES'):
            for key in cls._MATCHER_CLASSES:
                Matcher._METHODS[key] = cls

        elif register:
            key = cls.__name__.replace("Matcher", "").lower()

            for key_ in ALL_MATCHERS:
                if key_.replace("_", "") == key:
                    key = key_
                    break
            if key:
                Matcher._METHODS[key] = cls

    @staticmethod
    def create(matcher_name, logger, config, descriptor_class):
        matcher_name = matcher_name.lower()
        backend = config.get('backend') or get_default_backend(matcher_name, 2)
        backend = backend.lower() if backend is not None else None

        if matcher_name in DNN_PIPELINES:
            return Matcher._METHODS["dnnpipeline"](matcher_name, logger, config)

        if matcher_name in DNN_MATCHERS:
            return Matcher._METHODS['dnn'](matcher_name, logger, config, descriptor_class)

        if not Matcher._METHODS.get(matcher_name):
            raise ValueError(f"Matcher '{matcher_name}' not found."
                             f" Available: {list(Matcher._METHODS.keys())}")

        return Matcher._METHODS[matcher_name](matcher_name, logger, config, descriptor_class)

    @abstractmethod
    def match(self, features1, features2):
        pass

    @abstractmethod
    def _init_matcher(self):
        pass
