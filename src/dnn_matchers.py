from abc import abstractmethod

from src.matchers import Matcher
import src.backend.torch.model_wrapper
import src.backend.torch.inference_api
import src.backend.torch.io_adapter
from src.backend.io_adapter_base import IOAdapter
from src.backend.inference_api_base import InferenceAPI
from src.backend.model_wrapper_base import ModelWrapper


class DNNMatcher(Matcher):
    _model_wrapper = None
    _model_components = None
    _io_adapter = None
    _inference = None

    def __init__(self, matcher_name, logger, config=None, descriptor_class=None):
        if config is None:
            config = {}

        Matcher.__init__(self, matcher_name, logger, config, descriptor_class)

        backend = config.pop('backend', 'torch').lower()
        if 'descriptor_class' not in config:
            config['descriptor_class'] = descriptor_class

        self._nfeatures = config.get('nfeatures', 4096)
        self._threshold = config.get('threshold', 0.005)

        if DNNMatcher._model_wrapper is None:
            DNNMatcher._model_wrapper = ModelWrapper.create(backend=backend, model_name=matcher_name,
                                                            config=config, logger=logger)
            DNNMatcher._model_components = DNNMatcher._model_wrapper.load()

        if DNNMatcher._io_adapter is None:
            DNNMatcher._io_adapter = IOAdapter.create(backend=backend, model_name=matcher_name,
                                                      config=config, logger=logger)

        if DNNMatcher._inference is None:
            DNNMatcher._inference = InferenceAPI.create(backend=backend, logger=logger, model_name=matcher_name,
                                                        model_wrapper=DNNMatcher._model_wrapper, config=config)

    def _init_matcher(self):
        pass

    def match(self, features0, features1):
        inputs = {"features0": features0, "features1": features1, "descriptor_name": self._descriptor_class,
                  **DNNMatcher._model_components}
        inputs = DNNMatcher._io_adapter.preprocess(inputs)
        outputs = DNNMatcher._inference.run(inputs)
        outputs = DNNMatcher._io_adapter.postprocess(outputs)

        if outputs['matches'] is not None:
            self._logger.info(f"{self._matcher_name} match {len(outputs['matches'])} points")
        else:
            self._logger.warning(f"{self._matcher_name} match 0 points")
        return outputs

    def close(self):
        DNNMatcher._inference = None
        DNNMatcher._io_adapter = None
        DNNMatcher._model_components = None
        DNNMatcher._model_wrapper = None