import numpy as np
import torch

from src.backend.executorch.model_wrapper.model_wrapper import ExecuTorchModelWrapper
from src.backend.executorch.utils import prepare_image_tensor
from src.backend.model_wrapper_base import ModelWrapper


@ModelWrapper.register("aliked_lightglue_executorch")
class AlikedLightGlueExecuTorchModelWrapper(ExecuTorchModelWrapper):
    def call(self, inputs):
        image = inputs['image']
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = prepare_image_tensor(tensor, self._config.get('input_shape', (1, 3, 640, 640)))
        with torch.no_grad():
            points, descriptors, scores = super().call((tensor.contiguous().cpu(),))

        height, width = image.shape[:2]
        input_height, input_width = tensor.shape[-2:]
        points = points.detach().cpu().numpy().astype(np.float32, copy=True).reshape(-1, 2)
        points *= (width / input_width, height / input_height)
        return {'keypoints': points, 'descriptors': descriptors.squeeze(0).detach().cpu().numpy(),
                'scores': scores.detach().cpu().numpy().reshape(-1)}
