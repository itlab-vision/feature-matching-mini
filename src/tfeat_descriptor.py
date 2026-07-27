import cv2 as cv
import torch
import sys
from pathlib import Path

TFEAT_ROOT = str(Path(__file__).parent.parent / "tfeat")
if TFEAT_ROOT not in sys.path:
    sys.path.append(TFEAT_ROOT)

from src.descriptors import Descriptor  # noqa: E402
from tfeat.tfeat_model import TNet  # noqa: E402
from tfeat.tfeat_utils import describe_opencv  # noqa: E402


class TFeat(Descriptor):
    def __init__(self, descriptor_name, logger, config):
        super().__init__(logger, descriptor_name)
        self.model = TNet()
        base_dir = Path(__file__).parent.parent
        models_path = config.pop('tfeat_model_path',
                                 str(base_dir / "tfeat" / "pretrained-models" / "tfeat-liberty.params"))
        self.mag_factor = config.pop('magfactor', 3)
        device = config.pop('device', None)
        if device is None:
            if torch.cuda.is_available():
                self._device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self._device = torch.device('mps')
            else:
                self._device = torch.device('cpu')
        else:
            self._device = torch.device(device)

        state_dict = torch.load(models_path, map_location=self._device)
        self.model.load_state_dict(state_dict)
        self.model.to(self._device)
        self.model.eval()

    @property
    def default_norm(self):
        return cv.NORM_L2

    def compute(self, img, features):
        if img is None:
            self._logger.error("Input image is None. Detection aborted.")
            return {'kp': (), 'des': ()}

        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        self._logger.info(f"Running inference with {self._descriptor_name}")
        kp = features.get('kp')
        use_gpu = (self._device.type == 'cuda')
        desc_tfeat = describe_opencv(self.model, img, kp, 32, self.mag_factor, use_gpu=use_gpu)
        self._logger.info(f"{self._descriptor_name} computed {len(desc_tfeat)} descriptors")
        return {'kp': kp, 'des': desc_tfeat}
