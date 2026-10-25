import torch
import torch.nn.functional as functional
import numpy as np

from src.backend.io_adapter_base import IOAdapter
from src.backend.torch.io_adapter.io_adapter import TorchIOAdapter
from src.backend.torch.utils import features_to_tensor, matches_to_cv


@IOAdapter.register("superglue_torch")
class SuperGlueTorchIOAdapter(TorchIOAdapter):
    def _preprocess_features(self, inputs):
        inputs = features_to_tensor(inputs)
        kps = inputs['keypoints']
        des = inputs['descriptors']
        scores = inputs['scores']

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
            'image': torch.empty(1, 1, inputs['height'], inputs['width']).to(self._device)
        }
        return data

    def preprocess(self, inputs):
        features0 = inputs['features0']
        features1 = inputs['features1']

        features0 = self._preprocess_features(features0)
        features1 = self._preprocess_features(features1)

        input_dict = {
            'keypoints0': features0['keypoints'],
            'keypoints1': features1['keypoints'],
            'descriptors0': features0['descriptors'],
            'descriptors1': features1['descriptors'],
            'scores0': features0['scores'],
            'scores1': features1['scores'],
            'image0': features0['image'],
            'image1': features1['image'],
        }
        return input_dict

    def postprocess(self, outputs):
        matches0 = outputs['matches'][0].cpu().numpy()
        confidences = outputs['scores'][0].cpu().numpy()

        num_keypoints1 = outputs['keypoints1'].shape[1]
        valid = (matches0 > -1) & (matches0 < num_keypoints1)
        idx0 = np.where(valid)[0]
        idx1 = matches0[valid]

        match_indices = np.stack([idx0, idx1], axis=1)
        res_scores = confidences[valid]

        matches_and_scores = {'matches': torch.from_numpy(match_indices).long(),
                              'scores': torch.from_numpy(res_scores).float()}
        match_cv = matches_to_cv(matches_and_scores)
        return match_cv
