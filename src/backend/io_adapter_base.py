from abc import ABC, abstractmethod


class IOAdapter(ABC):
    _METHODS = {}

    def __init__(self, model_name, model_wrapper, logger=None, config=None):
        if config is None:
            config = {}

        self._model_name = model_name.lower()
        self._config = config
        self._logger = logger
        self._model = model_wrapper

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
    def create(cls, backend, model_name, model_wrapper=None, logger=None, config=None):
        model_key = model_name.lower()
        adapter_class = cls._METHODS.get(model_key)

        if adapter_class is None:
            adapter_class = cls._METHODS.get(f"{model_key}_{backend.lower()}")
        if adapter_class is None:
            raise ValueError(f"IO adapter for model '{model_name}' on backend '{backend}' not found. "
                             f"Available: {list(cls._METHODS.keys())}")

        return adapter_class(model_name=model_name, model_wrapper=model_wrapper,
                             config=config, logger=logger)

    @abstractmethod
    def preprocess(self, inputs):
        pass

    @abstractmethod
    def postprocess(self, outputs):
        pass
