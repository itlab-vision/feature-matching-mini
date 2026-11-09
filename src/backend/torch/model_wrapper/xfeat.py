import torch
import cv2 as cv

from thirdparty.xfeat.modules.xfeat import XFeat as XFeatModel

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.torch.model_wrapper.model_wrapper import TorchModelWrapper


@ModelWrapper.register("xfeat_torch")
class XFeatTorchModelWrapper(TorchModelWrapper):
    def load(self):
        self._logger.info(f"Loading XFeat weights onto {self._device}")
        self._model = XFeatModel().to(self._device)
        self._model.dev = self._device
        self._model.eval()
        return {'model': self._model}

    def preprocess(self, inputs):
        image = inputs["image"]
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        return torch.from_numpy(image_rgb).permute(2, 0, 1).float().unsqueeze(0).to(self._device) / 255.0

    @staticmethod
    def postprocess(output):
        return {key: value.detach().cpu().numpy() if torch.is_tensor(value) else value
                for key, value in output.items()}

    def call(self, inputs):
        try:
            image = self.preprocess(inputs)
            with torch.no_grad():
                output = self._model.detectAndCompute(image, top_k=self._nfeatures)[0]
            return self.postprocess(output)
        except Exception as error:
            self._logger.warning(f"XFeat inference error: {error}")
            raise
