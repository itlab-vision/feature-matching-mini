import sys
import torch
import cv2 as cv
import numpy as np
from pathlib import Path
from PIL import Image
from src.dnn_pipeline import DNNPipeline

ROMA_ROOT = Path(__file__).resolve().parent.parent / '3rdparty' / 'roma'
if str(ROMA_ROOT) not in sys.path:
    sys.path.insert(0, str(ROMA_ROOT))

from romatch import roma_outdoor  # noqa: E402


class RoMa(DNNPipeline):
    def __init__(self, extractor_name, logger, config=None, descriptor_name=None):
        if config is None:
            config = {}

        DNNPipeline.__init__(self, extractor_name, logger, config)

        self._num_features = config.get('num_features', 4096)
        self._coarse_res = config.get('coarse_res', 560)
        self._upsample_res = config.get('upsample_res', (864, 1152))

        if RoMa._model is None:
            try:
                self._logger.info(f"Loading RoMa weights onto {self._device}")
                RoMa._model = roma_outdoor(device=self._device, coarse_res=self._coarse_res ,
                                           upsample_res=self._upsample_res)
                RoMa._model.eval().to(self._device)
                self._logger.info("RoMa loaded successfully.")
            except Exception as e:
                self._logger.error(f"Failed to load RoMa: {e}")

        self._model = RoMa._model

    def _preprocess(self, img):
        if torch.is_tensor(img):
            if img.ndim == 4:
                img = img.squeeze(0)
            img_np = img.permute(1, 2, 0).cpu().detach().numpy()

            if img_np.max() <= 1.01:
                img_np = (img_np * 255).astype(np.uint8)
            return Image.fromarray(img_np)
        img_np = np.array(img)

        if len(img_np.shape) == 3:
            img_rgb = cv.cvtColor(img_np.astype(np.uint8), cv.COLOR_BGR2RGB)
        else:
            img_rgb = cv.cvtColor(img_np.astype(np.uint8), cv.COLOR_GRAY2RGB)
        return Image.fromarray(img_rgb)

    def match(self, features0, features1):
        img0_raw = features0.get('image')
        img1_raw = features1.get('image')

        if img0_raw is None or img1_raw is None:
            self._logger.error("Input image is None. Detection aborted.")
            return {'matches': (), 'scores': ()}

        pil0 = self._preprocess(img0_raw)
        pil1 = self._preprocess(img1_raw)

        w0_orig, h0_orig = pil0.size
        w1_orig, h1_orig = pil1.size

        try:
            with torch.no_grad():
                warp, certainty = self._model.match(pil0, pil1, device=self._device)
                matches, conf = self._model.sample(warp, certainty, num=self._num_features)

                kp0, kp1 = self._model.to_pixel_coordinates(matches, h0_orig, w0_orig, h1_orig, w1_orig)

            self._logger.info(f"RoMa found {kp0.shape[0]} valid matches.")

            indices = torch.arange(len(kp0)).view(-1, 1).repeat(1, 2)

            return {
                'keypoints0': kp0.cpu(),
                'keypoints1': kp1.cpu(),
                'matches': indices.long(),
                'scores': conf.cpu()
            }

        except Exception as e:
            self._logger.error(f"RoMa match error: {e}")
            return {'matches': (), 'scores': ()}
