import torch
import torch.nn.functional as F
import numpy as np

from src.backend.executorch.utils import prepare_image_tensor, nms_topk, sample_descriptors

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.executorch.model_wrapper.model_wrapper import ExecuTorchModelWrapper


@ModelWrapper.register("xfeat_executorch")
class XFeatExecuTorchModelWrapper(ExecuTorchModelWrapper):
    def load(self):
        return super().load()

    def preprocess(self, inputs):
        if isinstance(inputs, dict):
            image = inputs.get("image")
        else:
            image = inputs

        if not isinstance(image, np.ndarray):
            raise TypeError("XFeat ExecuTorch expects a NumPy image")

        original_size = image.shape[:2]
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = prepare_image_tensor(tensor, self._config.get("input_shape", (1, 3, 480, 640)))
        input_size = tensor.shape[-2:]
        return tensor, (original_size, input_size)

    def postprocess(self, outputs, sizes):
        original_size, input_size = sizes
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
            raise ValueError("Expected XFeat outputs (logits, dense_descriptors)")

        logits, dense = outputs
        if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 65:
            raise ValueError("Expected XFeat logits with shape [1, 65, H, W]")

        if dense.ndim != 4 or dense.shape[0] != 1 or dense.shape[-2:] != logits.shape[-2:]:
            raise ValueError("Expected matching XFeat descriptor map")

        probabilities = F.softmax(logits[:, :-1], dim=1)
        _, _, coarse_height, coarse_width = probabilities.shape
        scores = probabilities.permute(0, 2, 3, 1).reshape(1, coarse_height, coarse_width, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(coarse_height * 8, coarse_width * 8)

        count = min(self._config.get("num_keypoints", self._config.get("nfeatures", 4096)), scores.numel())
        keypoints, values = nms_topk(scores, count, self._config.get("nms_radius", 4))
        descriptors = sample_descriptors(dense, keypoints / 8, coarse_height, coarse_width)
        keypoints[:, 0] *= original_size[1] / input_size[1]
        keypoints[:, 1] *= original_size[0] / input_size[0]
        return {"keypoints": keypoints.detach().cpu().numpy().astype(np.float32, copy=False),
                "descriptors": descriptors.detach().cpu().numpy().astype(np.float32, copy=False),
                "scores": values.detach().cpu().numpy().astype(np.float32, copy=False)}

    def call(self, inputs):
        if self._method is None:
            raise RuntimeError("ExecuTorch model is not loaded. Call load() first.")

        try:
            tensor, sizes = self.preprocess(inputs)
            with torch.no_grad():
                outputs = self._method.execute((tensor.contiguous().cpu(),))
            return self.postprocess(outputs, sizes)

        except Exception as error:
            self._logger.error(f"XFeat inference error: {error}")
            raise
