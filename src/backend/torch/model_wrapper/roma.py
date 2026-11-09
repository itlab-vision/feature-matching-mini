import torch
import cv2 as cv
import numpy as np
from PIL import Image

from thirdparty.roma.romatch import roma_outdoor

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.torch.model_wrapper.model_wrapper import TorchModelWrapper


@ModelWrapper.register("roma_torch")
class RoMaTorchModelWrapper(TorchModelWrapper):
    def load(self):
        coarse_res = self._config.get('coarse_res', 560)
        upsample_res = self._config.get('upsample_res', (864, 1152))
        use_custom_corr = self._config.get('use_custom_corr', False)

        self._logger.info(f"Loading RoMa weights onto {self._device}")
        self._model = roma_outdoor(device=self._device, coarse_res=coarse_res, upsample_res=upsample_res,
                                   use_custom_corr=use_custom_corr)
        self._model.eval().to(self._device)
        return {'model': self._model}

    def preprocess(self, inputs):
        images = []
        sizes = []
        for key in ('image0', 'image1'):
            image = np.asarray(inputs[key]).astype(np.uint8)
            color = cv.COLOR_BGR2RGB if image.ndim == 3 else cv.COLOR_GRAY2RGB
            images.append(Image.fromarray(cv.cvtColor(image, color)))
            sizes.append(image.shape[:2])
        return images, sizes

    @staticmethod
    def postprocess(outputs):
        keypoints0, keypoints1, scores = outputs
        points0 = keypoints0.detach().cpu().numpy()
        points1 = keypoints1.detach().cpu().numpy()
        indices = np.arange(len(points0), dtype=np.int64)
        return {'keypoints0': points0, 'keypoints1': points1,
                'matches': np.column_stack((indices, indices)),
                'scores': scores.detach().cpu().numpy()}

    def call(self, inputs):
        try:
            images, sizes = self.preprocess(inputs)
            with torch.no_grad():
                warp, certainty = self._model.match(*images, device=self._device)
                matches, conf = self._model.sample(warp, certainty, num=self._nfeatures)
                keypoints = self._model.to_pixel_coordinates(matches, *sizes[0], *sizes[1])
            return self.postprocess((*keypoints, conf))
        except Exception as error:
            self._logger.error(f"RoMa match error: {error}")
            raise
