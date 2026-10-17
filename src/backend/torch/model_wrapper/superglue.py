import torch

from thirdparty.superglue.models.superglue import SuperGlue

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.torch.model_wrapper.model_loader import TorchModelWrapper


@ModelWrapper.register("superglue_torch")
class SuperGlueTorchModelWrapper(TorchModelWrapper):
    def load(self):
        sg_config = {
            'weights': self._config.get('weights', 'outdoor'),
            'sinkhorn_iterations': self._config.get('sinkhorn_iterations', 20),
            'match_threshold': self._config.get('threshold', 0.005),
        }

        self._logger.info(f"Loading SuperGlue ({sg_config.get('weights')}) onto {self._device}")
        self._model = SuperGlue(sg_config).to(self._device).eval()
        return {"model": self._model}

    def call(self, inputs):
        try:
            with torch.no_grad():
                outputs = self._model(inputs)

            matches = outputs['matches0']
            scores = outputs['matching_scores0']
            return {'matches' : matches, 'scores' : scores, **inputs}

        except Exception as e:
            self._logger.error(f"Super Glue inference error: {e}")
            raise
