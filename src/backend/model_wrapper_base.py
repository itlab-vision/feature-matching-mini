from abc import ABC, abstractmethod


class ModelWrapper(ABC):
    _METHODS = {}

    def __init__(self, model_name, model_path=None, config=None, logger=None):
        if config is None:
            config = {}

        self._model_name = model_name.lower()
        self._model_path = model_path
        self._config = config
        self._logger = logger
        self._model = None

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            aliases = getattr(subclass, '_EXTRACTOR_CLASSES', {})

            for key in aliases:
                cls._METHODS[key.lower()] = subclass

            cls._METHODS[name.lower()] = subclass

            return subclass

        return decorator

    @classmethod
    def create(cls, backend, model_name, model_path=None, config=None, logger=None):
        loader_name = f"{model_name.lower()}_{backend.lower()}"
        if loader_name not in cls._METHODS:
            raise ValueError(f"Model loader backend '{backend}' not found. "
                             f"Available: {list(cls._METHODS.keys())}")

        return cls._METHODS[loader_name](model_name=model_name, model_path=model_path, config=config, logger=logger)

    @abstractmethod
    def load(self):
        pass

    @property
    def model(self):
        return self._model
