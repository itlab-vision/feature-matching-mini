import torch

from thirdparty.xfeat.modules.xfeat import XFeat as XFeatModel

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.torch.model_wrapper.model_loader import TorchModelWrapper


@ModelWrapper.register("xfeat_torch")
class XFeatTorchModelWrapper(TorchModelWrapper):
    def load(self):
        self._logger.info(f"Loading XFeat weights onto {self._device}")
        self._model = XFeatModel().to(self._device)
        self._model.dev = self._device
        self._model.eval()
        return {'model': self._model}

    def call(self, inputs):
        img = inputs.get("image")

        try:
            with torch.no_grad():
                output = self._model.detectAndCompute(img, top_k=self._nfeatures)[0]

            return {'keypoints': output['keypoints'],
                    'descriptors': output['descriptors'],
                    'scores': output['scores']}

        except Exception as e:
            self._logger.warning(f"XFeat inference error: {e}")
            raise
