import sys
import torch
import numpy as np
import cv2 as cv
from pathlib import Path

from src.dnn_extractors import DNNFeatureExtractors

R2D2_ROOT = Path(__file__).resolve().parent.parent / "3rdparty" / "r2d2"
if str(R2D2_ROOT) not in sys.path:
    sys.path.insert(0, str(R2D2_ROOT))

from extract import NonMaxSuppression, extract_multiscale, load_network  # noqa: E402


class R2D2(DNNFeatureExtractors):
    _nms = None

    def __init__(self, extractor_name, logger, config=None):
        config = config or {}

        DNNFeatureExtractors.__init__(self, extractor_name, logger, config)
        checkpoint = config.pop('checkpoint', "3rdparty/r2d2/models/r2d2_WASF_N16.pt")

        if R2D2._model is None:
            try:
                self._logger.info(f"Loading R2D2 weights from {checkpoint}")
                R2D2._model = load_network(checkpoint).to(self._device).eval()
                R2D2._nms = NonMaxSuppression()
            except Exception as e:
                self._logger.error(f"Failed to load R2D2: {e}")
                raise

        self._model = R2D2._model

    def _preprocess(self, img):
        if torch.is_tensor(img):
            img_np = img.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
            if img_np.max() <= 1.01:
                img_np *= 255.0
        else:
            img_np = np.array(img)

        img_rgb = cv.cvtColor(img_np.astype(np.uint8), cv.COLOR_BGR2RGB)

        input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        input_tensor = (input_tensor - mean) / std
        return input_tensor.unsqueeze(0).to(self._device)

    def _forward(self, img):
        if img is None:
            self._logger.error("Input image is None. Detection aborted.")
            return {'keypoints': (), 'descriptors': ()}

        self._logger.info(f"Running inference with {self._detector_name}")
        input_tensor = self._preprocess(img)

        try:
            with torch.no_grad():
                xys, desc, scores = extract_multiscale(self._model, input_tensor, R2D2._nms)

            mask = scores > self._threshold
            xys, desc, scores = xys[mask], desc[mask], scores[mask]

            if len(scores) > self._nfeatures:
                _, top_indices = torch.topk(scores, k=self._nfeatures)
                xys, desc, scores = xys[top_indices], desc[top_indices], scores[top_indices]

            extracted_data = {
                'keypoints': xys[:, :2].cpu(),
                'descriptors': desc.cpu(),
                'scores': scores.cpu()
            }
            R2D2._extracted_data = extracted_data

            if len(xys[:, :2]) > 0:
                self._logger.info(f"{self._detector_name} found {len(xys[:, :2])} points")
            else:
                self._logger.warning(f"{self._detector_name} found 0 points")

            if desc is not None:
                self._logger.info(f"{self._descriptor_name} computed {len(desc)} descriptors")
            else:
                self._logger.warning(f"{self._descriptor_name} computed 0 descriptors")

            self._logger.info(f"{self._detector_name} found {len(xys)} points")
            return extracted_data

        except Exception as e:
            self._logger.warning(f"{self._detector_name} inference failed (likely 0 points): {e}")
            return {'keypoints': (), 'descriptors': ()}
