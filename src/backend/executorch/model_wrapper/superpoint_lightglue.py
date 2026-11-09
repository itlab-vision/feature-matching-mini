import torch
import numpy as np

from src.backend.executorch.utils import prepare_image_tensor, nms_topk, sample_descriptors

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.executorch.model_wrapper.model_wrapper import ExecuTorchModelWrapper


@ModelWrapper.register("superpoint_lightglue_executorch")
class SuperPointLightGlueExecuTorchModelWrapper(ExecuTorchModelWrapper):
    def load(self):
        return super().load()

    def preprocess(self, inputs):
        image = np.asarray(inputs['image'])
        original_size = image.shape[:2]
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = prepare_image_tensor(tensor, self._config.get('input_shape', (1, 3, 480, 640)))
        return tensor, (original_size, tensor.shape[-2:])

    def postprocess(self, outputs, sizes):
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
            raise ValueError('Expected SuperPoint outputs (logits, probabilities, dense_descriptors)')
        _, probabilities, dense = outputs
        if probabilities.ndim != 4 or probabilities.shape[1] != 65 or dense.shape[-2:] != probabilities.shape[-2:]:
            raise ValueError('Invalid SuperPoint output shapes')
        probabilities = probabilities[:, :-1]
        _, _, height, width = probabilities.shape
        scores = probabilities.permute(0, 2, 3, 1).reshape(1, height, width, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(height * 8, width * 8)
        count = min(self._config.get('num_keypoints', 256), scores.numel())
        keypoints, values = nms_topk(scores, count, self._config.get('nms_radius', 4))
        descriptors = sample_descriptors(dense, keypoints / 8, height, width)
        keypoints[:, 0] *= sizes[0][1] / sizes[1][1]
        keypoints[:, 1] *= sizes[0][0] / sizes[1][0]
        return {'keypoints': keypoints.detach().cpu().numpy().astype(np.float32),
                'descriptors': descriptors.detach().cpu().numpy().astype(np.float32),
                'scores': values.detach().cpu().numpy().astype(np.float32)}

    def call(self, inputs):
        if self._method is None:
            raise RuntimeError("ExecuTorch model is not loaded. Call load() first.")
        try:
            tensor, sizes = self.preprocess(inputs)
            with torch.no_grad():
                outputs = self._method.execute((tensor.contiguous().cpu(),))
            return self.postprocess(outputs, sizes)
        except Exception as error:
            if self._logger:
                self._logger.warning(f"SuperPoint ExecuTorch inference error: {error}")
            raise
