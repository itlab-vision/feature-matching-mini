import cv2 as cv
import torch
import sys
import numpy as np
from pathlib import Path

HARDNET_ROOT = Path(__file__).parent.parent / "3rdparty"
if str(HARDNET_ROOT) not in sys.path:
    sys.path.append(str(HARDNET_ROOT))

from src.descriptors import Descriptor  # noqa: E402
from hardnet.examples.extract_hardnet_desc_from_hpatches_file import HardNet as HardNetModel  # noqa: E402


class HardNet(Descriptor):
    MEAN = 0.443728476019
    STD = 0.20197947209

    def __init__(self, descriptor_name, logger, config):
        super().__init__(logger, descriptor_name)
        model_path = config.pop('hardnet_model_path',
                                str(HARDNET_ROOT / "hardnet" / "pretrained"
                                    / "train_liberty_with_aug" / "checkpoint_liberty_with_aug.pth"))
        self.patch_size = config.pop('patch_size', 32)
        self.batch_size = config.pop('batch_size', 128)
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

        self.model = HardNetModel()
        checkpoint = torch.load(model_path, map_location=self._device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
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

        kp = features.get('kp')
        if kp is None or len(kp) == 0:
            self._logger.info("No keypoints provided. Returning empty descriptors.")
            return {'kp': (), 'des': np.empty((0, 128), dtype=np.float32)}

        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        self._logger.info(f"Running inference with {self._descriptor_name}")
        keypoints, descriptors = self._extract_descriptors(img, kp)
        self._logger.info(f"{self._descriptor_name} computed {len(descriptors)} descriptors")
        return {'kp': keypoints, 'des': descriptors}

    def _extract_descriptors(self, img, kps):
        h, w = img.shape
        half = self.patch_size // 2
        valid_patches = []
        valid_kps = []
        for kp in kps:
            x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
            if x - half < 0 or x + half >= w or y - half < 0 or y + half >= h:
                continue
            patch = img[y - half:y + half, x - half:x + half]
            valid_patches.append(patch)
            valid_kps.append(kp)

        if not valid_patches:
            return [], np.empty((0, 128), dtype=np.float32)

        tensor = torch.from_numpy(np.array(valid_patches)).float().unsqueeze(1) / 255.0
        tensor -= self.MEAN
        tensor /= self.STD
        tensor = tensor.to(self._device)

        all_descs = []
        with torch.no_grad():
            for i in range(0, len(tensor), self.batch_size):
                batch = tensor[i:min(i + self.batch_size, len(tensor))]
                out = self.model(batch)
                all_descs.append(out.cpu().numpy())
        return valid_kps, np.concatenate(all_descs, axis=0).astype(np.float32)
