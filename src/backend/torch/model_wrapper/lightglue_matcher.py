import torch

from thirdparty.lightglue.lightglue import LightGlue

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.torch.model_wrapper.model_wrapper import TorchModelWrapper


@ModelWrapper.register("lightglue_torch")
class LightGlueTorchModelWrapper(TorchModelWrapper):
    def load(self):
        extractor_name = self._config.get('descriptor_class')._descriptor_name.replace("_lightglue", "")
        self._logger.info(f"Loading SuperGlue onto {self._device}")
        self._model = LightGlue(features=extractor_name, **self._config).eval().to(self._device)
        return {"model": self._model}

    def call(self, inputs):
        try:
            with torch.no_grad():
                outputs = self._model(inputs)
            return {"matches": outputs["matches"], "scores": outputs["scores"]}

        except Exception as e:
            self._logger.error(f"LightGlue inference error: {e}")
            raise
