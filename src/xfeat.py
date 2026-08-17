import cv2 as cv
import torch
import sys
from pathlib import Path

XFEAT_ROOT = Path(__file__).resolve().parent.parent / "3rdparty" / "xfeat"
if str(XFEAT_ROOT) not in sys.path:
    sys.path.insert(0, str(XFEAT_ROOT))

from src.dnn_extractors import DNNFeatureExtractors  # noqa: E402
from src.image_utils import to_numpy_bgr  # noqa: E402

from modules.xfeat import XFeat as XFeatModel  # noqa: E402


class XFeat(DNNFeatureExtractors):
    def __init__(self, extractor_name, logger, config=None):
        if config is None:
            config = {}

        DNNFeatureExtractors.__init__(self, extractor_name, logger, config)

        self._top_k = config.pop('top_k', 4096)
        if config:
            self._logger.warning(f"XFeat: unknown config keys ignored: {list(config.keys())}")

        if self._device == torch.device('mps'):
            self._device = torch.device('cpu')

        if XFeat._model is None:
            try:
                self._logger.info(f"Loading XFeat weights onto {self._device}")
                XFeat._model = XFeatModel().to(self._device)
                XFeat._model.dev = self._device
                XFeat._model.eval()
            except Exception as e:
                self._logger.error(f"Failed to load XFeat: {e}")

    def _forward(self, img):
        if img is None:
            self._logger.error("Input image is None. Detection aborted.")
            return {'keypoints': (), 'descriptors': ()}

        self._logger.info(f"Running inference with {self._detector_name}")

        input_type = 'torch' if isinstance(img, torch.Tensor) else 'numpy'
        img_np = to_numpy_bgr(img, input_type=input_type)
        img_rgb = cv.cvtColor(img_np, cv.COLOR_BGR2RGB)

        input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0)
        input_tensor = input_tensor.to(self._device) / 255.0

        try:
            with torch.no_grad():
                output = self._model.detectAndCompute(input_tensor, top_k=self._top_k)[0]

            raw_kp = output['keypoints'].cpu()
            raw_des = output['descriptors'].cpu()
            raw_scores = output['scores'].cpu()

            mask = raw_scores > self._threshold

            extracted = {
                'keypoints': raw_kp[mask],
                'descriptors': raw_des[mask],
                'scores': raw_scores[mask].cpu().numpy()
            }
            XFeat._extracted_data = extracted

            if len(raw_kp[mask]) > 0:
                self._logger.info(f"{self._detector_name} found {len(raw_kp[mask])} points")
            else:
                self._logger.warning(f"{self._detector_name} found 0 points")

            if raw_des[mask] is not None:
                self._logger.info(f"{self._descriptor_name} computed {len(raw_des[mask])} descriptors")
            else:
                self._logger.warning(f"{self._descriptor_name} computed 0 descriptors")

            return extracted

        except Exception as e:
            self._logger.warning(f"{self._detector_name} inference failed (likely 0 points): {e}")
            return {'keypoints': (), 'descriptors': (), 'scores': ()}
