from pathlib import Path
import torch

from thirdparty.d2net.lib.model_test import D2Net as D2NetModel
from thirdparty.d2net.lib.pyramid import process_multiscale

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.torch.model_wrapper.model_wrapper import TorchModelWrapper
from src.backend.torch.model_wrapper.weights_url import WEIGHTS_URL_MODELS


@ModelWrapper.register("d2net_torch")
class D2NetTorchModelWrapper(TorchModelWrapper):
    def load(self):
        checkpoint = Path(self._model_path or self._config.pop('checkpoint', "weights/d2net/d2_tf.pth"))
        weights_url = WEIGHTS_URL_MODELS.get(self._model_name)

        use_relu = self._config.get('use_relu', True)
        device_str = self._config.get('device', 'cpu').lower()
        use_cuda = 'cuda' in device_str

        if not checkpoint.exists():
            self._logger.info(f"Downloading D2Net weights to {checkpoint}")
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(weights_url, str(checkpoint))

        self._logger.info(f"Initializing D2NetModel from {checkpoint}")
        self._model = D2NetModel(model_file=str(checkpoint), use_relu=use_relu, use_cuda=use_cuda)
        self._model = self._model.to(self._device)
        self._model.eval()
        return {'model': self._model}

    def call(self, inputs):
        img = inputs.get("image")

        try:
            with torch.no_grad():
                keypoints, scores, descriptors = process_multiscale(
                    torch.from_numpy(img).float().unsqueeze(0).to(self._device),
                    self._model, scales=[1])

            return {'keypoints': keypoints, 'descriptors': descriptors, 'scores': scores}

        except Exception as e:
            self._logger.error(f"D2-Net inference error: {e}")
            raise
