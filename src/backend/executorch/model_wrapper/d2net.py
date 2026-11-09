import torch
import torch.nn.functional as F
import numpy as np

from thirdparty.d2net.lib.model_test import HardDetectionModule, HandcraftedLocalizationModule

from src.backend.executorch.utils import prepare_image_tensor
from src.backend.model_wrapper_base import ModelWrapper
from src.backend.executorch.model_wrapper.model_wrapper import ExecuTorchModelWrapper


@ModelWrapper.register("d2net_executorch")
class D2NetExecuTorchModelWrapper(ExecuTorchModelWrapper):
    def __init__(self, model_name, model_path=None, config=None, logger=None):
        super().__init__(model_name, model_path, config, logger)

        self._detection = HardDetectionModule()
        self._localization = HandcraftedLocalizationModule()
        self._num_keypoints = self._config.get("num_keypoints", self._config.get("nfeatures", 4096))

    def load(self):
        return super().load()

    def preprocess(self, inputs):
        image = inputs.get("image") if isinstance(inputs, dict) else inputs
        if not isinstance(image, np.ndarray):
            raise TypeError("D2-Net ExecuTorch expects a NumPy image")

        original_size = image.shape[:2]
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = prepare_image_tensor(tensor, self._config.get("input_shape", (1, 3, 480, 640)))

        processed_size = tensor.shape[-2:]
        mean = tensor.new_tensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1)
        tensor = tensor[:, [2, 1, 0]].mul(255.0).sub(mean)
        return tensor, (original_size, processed_size)

    def postprocess(self, outputs, sizes):
        original_size, processed_size = sizes
        if isinstance(outputs, (tuple, list)):
            dense = outputs[0]
        else:
            dense = outputs

        if not isinstance(dense, torch.Tensor) or dense.ndim != 4 or dense.shape[0] != 1:
            raise ValueError("Expected D2-Net dense output with shape [1, C, H, W]")

        keypoints, descriptors, scores = self._extract_features(dense)
        keypoints[:, 0] *= original_size[1] / processed_size[1]
        keypoints[:, 1] *= original_size[0] / processed_size[0]
        return {"keypoints": keypoints.numpy().astype(np.float32, copy=False),
                "descriptors": descriptors.numpy().astype(np.float32, copy=False),
                "scores": scores.numpy().astype(np.float32, copy=False)}

    def call(self, inputs):
        if self._method is None:
            raise RuntimeError("ExecuTorch model is not loaded. Call load() first")

        try:
            tensor, sizes = self.preprocess(inputs)
            with torch.no_grad():
                outputs = self._method.execute((tensor.contiguous().cpu(),))
            return self.postprocess(outputs, sizes)
        except Exception as error:
            self._logger.error(f"D2-Net inference error: {error}")
            raise

    def _extract_features(self, dense):
        dense = dense.detach().cpu()
        detections = self._detection(dense)
        feature_positions = torch.nonzero(detections[0]).t()

        if feature_positions.shape[1] == 0:
            return self._empty_features(dense.shape[1])

        displacements = self._localization(dense)[0]
        displacement_i = displacements[0, feature_positions[0], feature_positions[1], feature_positions[2]]
        displacement_j = displacements[1, feature_positions[0], feature_positions[1], feature_positions[2]]
        valid = (displacement_i.abs() < 0.5) & (displacement_j.abs() < 0.5)

        feature_positions = feature_positions[:, valid]
        if feature_positions.shape[1] == 0:
            return self._empty_features(dense.shape[1])

        feature_keypoints = feature_positions[1:].float()
        feature_keypoints += torch.stack((displacement_i[valid], displacement_j[valid]))
        descriptors, valid_ids = self._interpolate_descriptors(feature_keypoints, dense[0])

        feature_positions = feature_positions[:, valid_ids]
        feature_keypoints = feature_keypoints[:, valid_ids]
        scores = dense[0, feature_positions[0], feature_positions[1], feature_positions[2]]

        keypoints = feature_keypoints * 4.0 + 1.5
        keypoints = keypoints[[1, 0]].t().contiguous()
        descriptors = F.normalize(descriptors, p=2, dim=0).t().contiguous()

        if self._num_keypoints is not None and scores.numel() > self._num_keypoints:
            scores, indices = torch.topk(scores, self._num_keypoints)
            keypoints = keypoints[indices]
            descriptors = descriptors[indices]

        return keypoints, descriptors, scores

    @staticmethod
    def _interpolate_descriptors(keypoints, dense):
        height, width = dense.shape[-2:]
        rows, columns = keypoints

        top = torch.floor(rows).long()
        left = torch.floor(columns).long()
        bottom = torch.ceil(rows).long()
        right = torch.ceil(columns).long()
        valid = (top >= 0) & (left >= 0) & (bottom < height) & (right < width)
        valid_ids = torch.nonzero(valid).flatten()

        if valid_ids.numel() == 0:
            return dense.new_empty((dense.shape[0], 0)), valid_ids

        rows = rows[valid_ids]
        columns = columns[valid_ids]
        top = top[valid_ids]
        left = left[valid_ids]
        bottom = bottom[valid_ids]
        right = right[valid_ids]

        row_offset = rows - top.float()
        column_offset = columns - left.float()
        descriptors = (
            (1.0 - row_offset) * (1.0 - column_offset) * dense[:, top, left]
            + (1.0 - row_offset) * column_offset * dense[:, top, right]
            + row_offset * (1.0 - column_offset) * dense[:, bottom, left]
            + row_offset * column_offset * dense[:, bottom, right]
        )
        return descriptors, valid_ids

    @staticmethod
    def _empty_features(descriptor_size):
        return (
            torch.empty((0, 2), dtype=torch.float32),
            torch.empty((0, descriptor_size), dtype=torch.float32),
            torch.empty(0, dtype=torch.float32),
        )
