import torch
import numpy as np

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.executorch.model_wrapper.model_wrapper import ExecuTorchModelWrapper


@ModelWrapper.register("lightglue_disk_executorch")
@ModelWrapper.register("lightglue_superpoint_executorch")
@ModelWrapper.register("lightglue_executorch")
class LightGlueExecuTorchModelWrapper(ExecuTorchModelWrapper):
    def load(self):
        return super().load()

    def preprocess(self, inputs):
        count = self._config.get('num_keypoints', 256)
        tensors = []
        for key in ('keypoints', 'descriptors'):
            for index in (0, 1):
                value = np.asarray(inputs[f'features{index}'][key], dtype=np.float32)
                if len(value) != count:
                    raise ValueError(f'LightGlue ExecuTorch expects {count} {key} for features{index}')
                tensors.append(torch.from_numpy(np.ascontiguousarray(value)).unsqueeze(0))
        return tuple(tensors)

    @staticmethod
    def postprocess(outputs):
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 4:
            raise ValueError('Expected LightGlue outputs (matches0, matches1, scores0, scores1)')
        matches, _, scores, _ = outputs
        indices = torch.arange(matches.shape[1])
        valid = (matches[0] >= 0) & (matches[0] < matches.shape[1])
        return {'matches': torch.stack((indices[valid], matches[0][valid].long()), dim=1).numpy(),
                'scores': scores[0][valid].detach().cpu().numpy()}

    def call(self, inputs):
        if self._method is None:
            raise RuntimeError("ExecuTorch model is not loaded. Call load() first.")
        try:
            tensors = self.preprocess(inputs)
            with torch.no_grad():
                outputs = self._method.execute(tensors)
            return self.postprocess(outputs)
        except Exception as error:
            if self._logger:
                self._logger.warning(f"LightGlue ExecuTorch inference error: {error}")
            raise
