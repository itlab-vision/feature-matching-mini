import torch
import numpy as np
import sys
from pathlib import Path

from src.dnn_extractors import DNNFeatureExtractors

D2_ROOT = Path(__file__).resolve().parent.parent / "3rdparty" / "d2net"
if str(D2_ROOT) not in sys.path:
    sys.path.insert(0, str(D2_ROOT))

from lib.model_test import D2Net as D2NetModel  # noqa: E402
from lib.pyramid import process_multiscale  # noqa: E402
from lib.utils import preprocess_image  # noqa: E402


class D2Net(DNNFeatureExtractors):
    WEIGHTS_URL = "https://dusmanu.com/files/d2-net/d2_tf.pth"

    def __init__(self, extractor_name, logger, config=None):
        if config is None:
            config = {}

        DNNFeatureExtractors.__init__(self, extractor_name, logger, config)

        checkpoint = Path(config.pop('checkpoint', "weights/d2net/d2_tf.pth"))
        if not checkpoint.exists():
            self._logger.info(f"Downloading weights to {checkpoint}")
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(self.WEIGHTS_URL, str(checkpoint))

        self._use_relu = config.pop('use_relu', True)

        if D2Net._model is None:
            self._logger.info(f"Loading D2Net weights onto {self._device}")
            use_cuda = self._device.type == 'cuda'
            D2Net._model = D2NetModel(model_file=checkpoint, use_relu=self._use_relu, use_cuda=use_cuda)
            D2Net._model.eval()

        self._model = D2Net._model

    def _preprocess(self, img):
        if torch.is_tensor(img):
            img_np = img.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
            if img_np.max() <= 1.0:
                img_np = (img_np * 255)
        else:
            img_np = np.array(img)

        return preprocess_image(img_np, preprocessing='caffe')

    def _forward(self, img):
        if img is None:
            self._logger.error("Input image is None. Detection aborted.")
            return {'keypoints': (), 'descriptors': ()}

        self._logger.info(f"Running inference with {self._detector_name}")
        img_prep = self._preprocess(img)

        try:
            with torch.no_grad():
                keypoints, scores, descriptors = process_multiscale(
                    torch.from_numpy(img_prep).float().unsqueeze(0).to(self._device),
                    self._model, scales=[1])

            mask = scores > self._threshold
            kp = keypoints[mask]
            des = descriptors[mask]
            sc = scores[mask]

            if self._nfeatures is not None and len(kp) > self._nfeatures:
                top_indices = np.argsort(sc)[::-1][:self._nfeatures]

                kp = kp[top_indices]
                des = des[top_indices]
                sc = sc[top_indices]

            if len(kp) > 0:
                xy_coords = kp[:, [1, 0]].astype(np.float32)

                extracted = {
                    'keypoints': torch.from_numpy(xy_coords),
                    'descriptors': torch.from_numpy(des),
                    'scores': torch.from_numpy(sc)
                }
                D2Net._extracted_data = extracted

                if len(keypoints[mask]) > 0:
                    self._logger.info(f"{self._detector_name} found {len(kp)} points")
                else:
                    self._logger.warning(f"{self._detector_name} found 0 points")

                if descriptors[mask] is not None:
                    self._logger.info(f"{self._descriptor_name} computed {len(des)} descriptors")
                else:
                    self._logger.warning(f"{self._descriptor_name} computed 0 descriptors")

                return extracted
            else:
                return {'keypoints': (), 'descriptors': ()}

        except Exception as e:
            self._logger.error(f"D2-Net inference error: {e}")
            return {'keypoints': (), 'descriptors': ()}
