from pathlib import Path
import torch
import cv2 as cv
import numpy as np

from thirdparty.loftr.src.loftr import LoFTR as LoFTRModel
from thirdparty.loftr.src.loftr import default_cfg

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.torch.model_wrapper.model_wrapper import TorchModelWrapper
from src.backend.torch.model_wrapper.weights_url import WEIGHTS_URL_MODELS


@ModelWrapper.register("loftr_torch")
class LoFTRTorchModelWrapper(TorchModelWrapper):
    def load(self):
        self._weights_type = self._config.pop('weights', 'outdoor')
        checkpoint = Path(self._model_path
                          or self._config.pop('checkpoint', f"weights/loftr/loftr_{self._weights_type}.ckpt"))
        weights_url = WEIGHTS_URL_MODELS.get(self._model_name).get(self._weights_type)

        if not checkpoint.exists():
            self._logger.info(f"Downloading LoFTR {self._weights_type} weights to {checkpoint}")
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(weights_url, str(checkpoint))

        self._logger.info(f"Initializing LoFTR onto {self._device}")
        model = LoFTRModel(config=default_cfg)
        ckpt_data = torch.load(str(checkpoint), map_location='cpu')
        model.load_state_dict(ckpt_data['state_dict'])

        self._model = model.to(self._device).eval()
        self._logger.info("LoFTR successfully loaded.")
        return {'model': self._model}

    def _preprocess_image(self, image):
        if image.ndim == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        height, width = image.shape[:2]
        scale = min(1.0, 640 / max(height, width))

        new_height = max(8, int(height * scale) // 8 * 8)
        new_width = max(8, int(width * scale) // 8 * 8)

        if (height, width) != (new_height, new_width):
            image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)
        tensor = torch.from_numpy(np.ascontiguousarray(image)).float()
        if tensor.numel() and tensor.max() > 1.1:
            tensor = tensor / 255.0
        return tensor.unsqueeze(0).unsqueeze(0).to(self._device)

    def preprocess(self, inputs):
        return {'image0': self._preprocess_image(inputs['image0']),
                'image1': self._preprocess_image(inputs['image1'])}

    @staticmethod
    def postprocess(outputs):
        keypoints0 = outputs['mkpts0_f'].detach().cpu().numpy()
        keypoints1 = outputs['mkpts1_f'].detach().cpu().numpy()
        indices = np.arange(len(keypoints0), dtype=np.int64)
        return {'keypoints0': keypoints0, 'keypoints1': keypoints1,
                'matches': np.column_stack((indices, indices)),
                'scores': outputs['mconf'].detach().cpu().numpy()}

    def call(self, inputs):
        try:
            data = self.preprocess(inputs)
            with torch.no_grad():
                self._model(data)
            return self.postprocess(data)
        except Exception as error:
            self._logger.error(f"LoFTR inference error: {error}")
            raise
