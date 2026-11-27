import torch
import torch.nn.functional as functional
import numpy as np

from thirdparty.superglue.models.superglue import SuperGlue

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.torch.model_wrapper.model_wrapper import TorchModelWrapper


@ModelWrapper.register("superglue_torch")
class SuperGlueTorchModelWrapper(TorchModelWrapper):
    def load(self):
        sg_config = {
            'weights': self._config.get('weights', 'outdoor'),
            'sinkhorn_iterations': self._config.get('sinkhorn_iterations', 20),
            'match_threshold': self._config.get('threshold', 0.005),
        }

        self._logger.info(f"Loading SuperGlue ({sg_config.get('weights')}) onto {self._device}")
        self._model = SuperGlue(sg_config).to(self._device).eval()
        return {"model": self._model}

    def _preprocess_features(self, features):
        descriptors = torch.from_numpy(features['descriptors']).float()
        descriptors = functional.normalize(descriptors, p=2, dim=1)
        return {'keypoints': torch.from_numpy(features['keypoints']).float().unsqueeze(0).to(self._device),
                'descriptors': descriptors.T.unsqueeze(0).to(self._device),
                'scores': torch.from_numpy(features['scores']).float().unsqueeze(0).to(self._device),
                'image': torch.empty(1, 1, features['height'], features['width'], device=self._device)}

    def preprocess(self, inputs):
        features0 = self._preprocess_features(inputs['features0'])
        features1 = self._preprocess_features(inputs['features1'])
        return {f'{key}{index}': features[key]
                for index, features in enumerate((features0, features1))
                for key in ('keypoints', 'descriptors', 'scores', 'image')}

    @staticmethod
    def postprocess(outputs, inputs):
        matches = outputs['matches0'][0].detach().cpu().numpy()
        scores = outputs['matching_scores0'][0].detach().cpu().numpy()
        valid = (matches >= 0) & (matches < inputs['keypoints1'].shape[1])
        indices = np.flatnonzero(valid)
        return {'matches': np.column_stack((indices, matches[valid])).astype(np.int64),
                'scores': scores[valid]}

    def call(self, inputs):
        try:
            data = self.preprocess(inputs)
            with torch.no_grad():
                outputs = self._model(data)
            return self.postprocess(outputs, data)
        except Exception as error:
            self._logger.error(f"Super Glue inference error: {error}")
            raise
