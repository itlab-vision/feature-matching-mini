import numpy as np
from src.matchers import Matcher
from src.backend.io_adapter_base import IOAdapter
from src.backend.inference_api_base import InferenceAPI
from src.backend.model_wrapper_base import ModelWrapper


class LightGlueOpenCVMatcher(Matcher):
    def __init__(self, matcher_name, logger, config, descriptor):
        super().__init__(matcher_name, logger, config, descriptor)

        if descriptor._descriptor_name == 'aliked_opencv' and config.get('lightglue_model_path') is None:
            config['lightglue_model_path'] = "models/lightglue_for_aliked.onnx"

        self._loader = ModelWrapper.create(backend="opencv", model_name=matcher_name, config=config, logger=logger)
        self._model = self._init_matcher()
        self._io_adapter = IOAdapter.create(backend="opencv", model_name=matcher_name, model_wrapper=self._model,
                                            config=config, logger=logger)
        self._inference = InferenceAPI.create(backend="opencv", logger=logger, model_name=matcher_name,
                                              model_wrapper=self._model, config=config)

    def _init_matcher(self):
        return self._loader.load()

    def match(self, features1, features2):
        self._io_adapter.preprocess({'features1': features1, 'features2': features2})
        outputs = self._inference.run()
        outputs = self._io_adapter.postprocess(outputs)

        matches = outputs.get('matches', np.array([]))
        scores = outputs.get('scores', np.array([]))
        self._logger.info(f"LightGlue found {len(matches)} matches")

        return {'matches': matches, 'scores': scores}
