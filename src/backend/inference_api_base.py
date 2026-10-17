from abc import ABC, abstractmethod


class InferenceAPI(ABC):
    _METHODS = {}

    def __init__(self, logger, model_name, model_wrapper, config=None):
        if config is None:
            config = {}

        self._logger = logger
        self._model_name = model_name
        self._model_wrapper = model_wrapper
        self._config = config

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls._METHODS[name.lower()] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, backend, logger, model_name, model_wrapper , config=None):
        backend_key = backend.lower()

        if backend_key not in cls._METHODS:
            raise ValueError(f"Inference backend '{backend}' not found. "
                             f"Available: {list(cls._METHODS.keys())}")

        return cls._METHODS[backend_key](logger, model_name, model_wrapper, config)

    @abstractmethod
    def run(self, inputs):
        pass
