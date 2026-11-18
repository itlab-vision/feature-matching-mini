import torch
import numpy as np

from src.backend.executorch.utils import prepare_image_tensor, nms_topk, sample_descriptors

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.executorch.model_wrapper.model_wrapper import ExecuTorchModelWrapper


@ModelWrapper.register("disk_lightglue_executorch")
class DiskLightGlueExecuTorchModelWrapper(ExecuTorchModelWrapper):
    def load(self):
        return super().load()

    def preprocess(self, inputs):
        image = np.asarray(inputs['image'])
        original_size = image.shape[:2]
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = prepare_image_tensor(tensor, self._config.get('input_shape', (1, 3, 480, 640)))
        return tensor.mul(255.0), (original_size, tensor.shape[-2:])

    def postprocess(self, outputs, sizes):
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
            raise ValueError('DISK ExecuTorch model must return heatmap and dense descriptors')
        heatmap, dense = outputs
        if not isinstance(heatmap, torch.Tensor) or heatmap.ndim != 4 or heatmap.shape[:2] != (1, 1):
            raise ValueError('Expected DISK heatmap with shape [1, 1, H, W]')
        if not isinstance(dense, torch.Tensor) or dense.ndim != 4 or dense.shape[0] != 1:
            raise ValueError('Expected DISK descriptors with shape [1, C, H, W]')
        if dense.shape[-2:] != heatmap.shape[-2:]:
            raise ValueError('DISK heatmap and descriptors must have the same spatial shape')

        height, width = heatmap.shape[-2:]
        count = min(self._config.get('num_keypoints', 256), height * width)
        keypoints, scores = nms_topk(heatmap[0, 0], count, self._config.get('nms_radius', 4))
        descriptors = sample_descriptors(dense, keypoints, height, width)
        keypoints[:, 0] *= sizes[0][1] / sizes[1][1]
        keypoints[:, 1] *= sizes[0][0] / sizes[1][0]
        return {'keypoints': keypoints.detach().cpu().numpy().astype(np.float32, copy=False),
                'descriptors': descriptors.detach().cpu().numpy().astype(np.float32, copy=False),
                'scores': scores.detach().cpu().numpy().astype(np.float32, copy=False)}

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
                self._logger.warning(f"DISK ExecuTorch inference error: {error}")
            raise
