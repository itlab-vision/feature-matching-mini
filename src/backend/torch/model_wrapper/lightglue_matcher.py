import torch

from thirdparty.lightglue.lightglue import LightGlue

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.torch.model_wrapper.model_wrapper import TorchModelWrapper


@ModelWrapper.register("lightglue_torch")
class LightGlueTorchModelWrapper(TorchModelWrapper):
    def load(self):
        extractor_name = self._config.get('descriptor_class')._descriptor_name.replace("_lightglue", "")
        self._logger.info(f"Loading SuperGlue onto {self._device}")
        self._model = LightGlue(features=extractor_name, **self._config).eval().to(self._device)
        return {"model": self._model}

    def preprocess(self, inputs):
        result = {}
        for index in (0, 1):
            features = inputs[f'features{index}']
            data = {
                'keypoints': torch.from_numpy(features['keypoints']).float().unsqueeze(0).to(self._device),
                'descriptors': torch.from_numpy(features['descriptors']).float().unsqueeze(0).to(self._device),
            }
            if features.get('scores') is not None:
                data['keypoint_scores'] = torch.from_numpy(features['scores']).float().unsqueeze(0).to(self._device)
            for key in ('scales', 'oris', 'image_size'):
                if features.get(key) is not None:
                    data[key] = torch.as_tensor(features[key], device=self._device)
            result[f'image{index}'] = data
        return result

    @staticmethod
    def postprocess(outputs):
        matches = outputs['matches'][0].detach().cpu().numpy()
        scores = outputs['scores'][0].detach().cpu().numpy()
        return {'matches': matches, 'scores': scores}

    def call(self, inputs):
        try:
            data = self.preprocess(inputs)
            with torch.no_grad():
                outputs = self._model(data)
            return self.postprocess(outputs)
        except Exception as error:
            self._logger.error(f"LightGlue inference error: {error}")
            raise
