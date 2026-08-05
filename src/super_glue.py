import torch
import torch.nn.functional as functional
import sys
import numpy as np
from pathlib import Path

from src.dnn_matchers import DNNMatcher  # noqa: E402

SG_ROOT = Path(__file__).resolve().parent.parent / "3rdparty" / "superglue"
if str(SG_ROOT) not in sys.path:
    sys.path.insert(0, str(SG_ROOT))

from models.superglue import SuperGlue  # noqa: E402


class SuperGlueMatcher(DNNMatcher):
    def __init__(self, logger, matcher_name, descriptor_name, config=None):
        if config is None:
            config = {}

        DNNMatcher.__init__(self, logger, matcher_name, descriptor_name, config)
        sg_config = {
            'weights': config.pop('weights', 'outdoor'),
            'sinkhorn_iterations': config.pop('sinkhorn_iterations', 20),
            'match_threshold': config.pop('threshold', 0.005),
        }

        self.logger.info(f"Loading SuperGlue ({sg_config.get('weights')}) onto {self._device}")
        self._matcher = SuperGlue(sg_config).to(self._device).eval()

    def _init_matcher(self):
        pass

    def prep(self, feat):
        kps = feat['keypoints']
        des = feat['descriptors']
        scores = feat['scores']

        if not torch.is_tensor(kps):
            kps = torch.from_numpy(kps).float()
        if not torch.is_tensor(des):
            des = torch.from_numpy(des).float()
        if not torch.is_tensor(scores):
            scores = torch.from_numpy(scores).float()

        des = functional.normalize(des, p=2, dim=1)

        data = {
            'keypoints': kps.unsqueeze(0).to(self._device),
            'descriptors': des.T.unsqueeze(0).to(self._device),
            'scores': scores.unsqueeze(0).to(self._device),
        }
        data['image'] = torch.empty(1, 1, feat['height'], feat['width']).to(self._device)
        return data

    def match(self, features0, features1):
        if len(features0['keypoints']) == 0 or len(features1['keypoints']) == 0:
            return {'matches': (), 'scores': ()}

        data0 = self.prep(features0)
        data1 = self.prep(features1)

        input_dict = {
            'keypoints0': data0['keypoints'],
            'keypoints1': data1['keypoints'],
            'descriptors0': data0['descriptors'],
            'descriptors1': data1['descriptors'],
            'scores0': data0['scores'],
            'scores1': data1['scores'],
            'image0': data0['image'],
            'image1': data1['image'],
        }

        with torch.no_grad():
            pred = self._matcher(input_dict)

        matches0 = pred['matches0'][0].cpu().numpy()
        confidences = pred['matching_scores0'][0].cpu().numpy()

        valid = (matches0 > -1) & (matches0 < len(features1['keypoints']))
        idx0 = np.where(valid)[0]
        idx1 = matches0[valid]

        match_indices = np.stack([idx0, idx1], axis=1)
        res_scores = confidences[valid]

        self.logger.info(f"SuperGlue: {len(match_indices)} matches found.")

        return {
            'matches': torch.from_numpy(match_indices).long(),
            'scores': torch.from_numpy(res_scores).float()
        }
