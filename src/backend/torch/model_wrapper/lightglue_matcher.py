import torch

from thirdparty.lightglue.lightglue import LightGlue

from src.backend.lightglue_utils import scale_and_orientation
from src.backend.model_wrapper_base import ModelWrapper
from src.backend.torch.model_wrapper.model_wrapper import TorchModelWrapper


@ModelWrapper.register("lightglue_torch")
class LightGlueTorchModelWrapper(TorchModelWrapper):
    def load(self):
        extractor_name = self._config.get('descriptor_class')._descriptor_name.replace("_lightglue", "")
        self._logger.info(f"Loading LightGlue (features={extractor_name}) onto {self._device}")
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
            if inputs.get('add_scale_ori'):
                scales, oris = scale_and_orientation(features, len(features['keypoints']))
                data['scales'] = torch.as_tensor(scales, device=self._device)
                data['oris'] = torch.as_tensor(oris, device=self._device)
            image_size = features.get('image_size')
            if image_size is None and features.get('width') is not None and features.get('height') is not None:
                image_size = [[features['width'], features['height']]]
            if image_size is not None:
                data['image_size'] = torch.as_tensor(image_size, device=self._device)
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
