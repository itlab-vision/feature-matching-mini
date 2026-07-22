import cv2 as cv
import torch
import os
from src.descriptors import Descriptor
from tfeat.tfeat_model import TNet
from tfeat.tfeat_utils import describe_opencv


class TFeat(Descriptor):
    def __init__(self, descriptor_name, logger, config):
        super().__init__(logger, descriptor_name)
        self.model = TNet()
        models_path = config.pop('tfeat_model_path', "tfeat/pretrained-models")
        net_name = config.pop('tfeat_model_name', 'tfeat-liberty')
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

        self.model.load_state_dict(torch.load(os.path.join(models_path, net_name + ".params")))
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
