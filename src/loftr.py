import sys
import cv2 as cv
import torch
from pathlib import Path

from src.dnn_pipeline import DNNPipeline

LOFTR_ROOT = Path(__file__).resolve().parent.parent / "3rdparty" / "loftr" / "src"

if str(LOFTR_ROOT) not in sys.path:
    sys.path.insert(0, str(LOFTR_ROOT))

from loftr import LoFTR as LoFTRModel  # noqa: E402
from loftr import default_cfg  # noqa: E402


class LoFTR(DNNPipeline):
    WEIGHTS_URLS = {
        'outdoor': "http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_outdoor.ckpt",
        'indoor': "http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_indoor.ckpt"
    }

    def __init__(self, extractor_name, logger, config=None, descriptor_name=None):
        if config is None:
            config = {}

        DNNPipeline.__init__(self, extractor_name, logger, config)

        self._weights_type = config.pop('weights', 'outdoor')
        checkpoint = Path(config.pop('checkpoint', f"weights/loftr/loftr_{self._weights_type}.ckpt"))

        if not checkpoint.exists():
            self._logger.info(f"Downloading LoFTR {self._weights_type} weights to {checkpoint}")
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(self.WEIGHTS_URLS[self._weights_type], str(checkpoint))

        if config:
            self._logger.warning(f"LoFTR: unknown config keys ignored: {list(config.keys())}")

        if LoFTR._model is None:
            try:
                self._logger.info(f"Initializing LoFTR from {LOFTR_ROOT} onto {self._device}")

                model = LoFTRModel(config=default_cfg)
                ckpt_data = torch.load(str(checkpoint), map_location='cpu')
                model.load_state_dict(ckpt_data['state_dict'])

                LoFTR._model = model.to(self._device).eval()
                self._logger.info("LoFTR successfully loaded.")

            except Exception as e:
                self._logger.error(f"Failed to load from {checkpoint}: {e}")
                raise

        self._model = LoFTR._model

    def _preprocess(self, img):
        if torch.is_tensor(img):
            if img.ndim == 4:
                img = img.squeeze(0)

            if img.shape[0] == 3:
                img_np = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
                img_np = img_np.cpu().numpy()
            else:
                img_np = img.squeeze(0).cpu().numpy()
        else:
            if img.ndim == 3:
                img_np = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            else:
                img_np = img

        h, w = img_np.shape[:2]
        max_dim = 640
        scale = max_dim / max(h, w)
        if scale < 1.0:
            h, w = int(h * scale), int(w * scale)

        new_h, new_w = (h // 8) * 8, (w // 8) * 8

        if img_np.shape[0] != new_h or img_np.shape[1] != new_w:
            img_np = cv.resize(img_np, (new_w, new_h), interpolation=cv.INTER_AREA)

        tensor = torch.from_numpy(img_np).float()
        if tensor.max() > 1.1:
            tensor /= 255.0

        return tensor.unsqueeze(0).to(self._device)

    def match(self, features0, features1):
        img0_raw = features0.get('image')
        img1_raw = features1.get('image')

        if (img0_raw is None) or (img1_raw is None):
            self._logger.error("Input image is None. Detection aborted.")
            return {'matches': (), 'scores': ()}

        t0 = self._preprocess(img0_raw).unsqueeze(0)
        t1 = self._preprocess(img1_raw).unsqueeze(0)

        data = {"image0": t0, "image1": t1}

        try:
            with torch.no_grad():
                self._model(data)

            mkp0 = data['mkpts0_f']
            mkp1 = data['mkpts1_f']
            mconf = data['mconf']

            mask = mconf > self._threshold
            mkp0, mkp1, mconf = mkp0[mask], mkp1[mask], mconf[mask]

            num_matches = len(mkp0)
            indices = torch.arange(num_matches).view(-1, 1).repeat(1, 2)

            self._logger.info(f"LoFTR found {num_matches} matches")

            return {
                'keypoints0': mkp0.cpu(),
                'keypoints1': mkp1.cpu(),
                'matches': indices.long(),
                'scores': mconf.cpu()
            }

        except Exception as e:
            self._logger.error(f"LoFTR inference failed: {e}")
            return {'matches': (), 'scores': ()}
