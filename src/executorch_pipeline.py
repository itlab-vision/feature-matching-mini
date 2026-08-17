from pathlib import Path
import cv2 as cv
import torch
import numpy as np
import torch.nn.functional as functional
from executorch.runtime import Runtime, Verification

from src.algorithms import ALL_DETECTORS
from src.descriptors import Descriptor
from src.detectors import Detector
from src.matchers import Matcher


class ExecuTorch:
    @staticmethod
    def _load_method(model_path):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"ExecuTorch model does not exist: {path}")
        return Runtime.get().load_program(str(path), verification=Verification.Minimal).load_method("forward")

    @staticmethod
    def _nms_topk(scores, keypoints, radius):
        if scores.ndim != 2:
            raise ValueError(f"Expected an HxW score map, got {tuple(scores.shape)}")

        height, width = scores.shape
        if not 0 < keypoints <= height * width:
            raise ValueError(f"num_keypoints must be in [1, {height * width}], got {keypoints}")

        local_max = functional.max_pool2d(scores[None, None], kernel_size=radius * 2 + 1,
                                          stride=1, padding=radius)[0, 0]
        filtered = scores.masked_fill(scores != local_max, float("-inf"))
        values, indices = torch.topk(filtered.reshape(-1), keypoints)
        xy = torch.stack((indices % width, torch.div(indices, width, rounding_mode="floor")), dim=1)
        return xy.to(torch.float32), values

    @staticmethod
    def _sample_descriptors(dense, keypoints, image_height, image_width):
        _, channels, height, width = dense.shape
        x = keypoints[:, 0] / max(image_width - 1, 1) * 2 - 1
        y = keypoints[:, 1] / max(image_height - 1, 1) * 2 - 1

        grid = torch.stack((x, y), dim=-1).reshape(1, 1, -1, 2)
        descriptors = functional.grid_sample(dense, grid, align_corners=True)[0, :, 0].transpose(0, 1)
        return functional.normalize(descriptors, p=2, dim=1)


class ExecuTorchDetector(ExecuTorch, Detector, register=False):
    def __init__(self, detector_name, logger, config=None):
        if config is None:
            config = {}
        Detector.__init__(self, logger, detector_name)

        model_path = config.get("executorch_model_path", config.get("model_path", None))
        if model_path is None:
            raise ValueError("ExecuTorch detector requires 'executorch_model_path' in detector config")

        self._method = self._load_method(model_path)
        self._input_shape = config.get("input_shape", (1, 3, 480, 640))
        self._num_keypoints = config.get("num_keypoints", 256)
        self._nms_radius = config.get("nms_radius", 4)

        if len(self._input_shape) != 4 or self._input_shape[0] != 1:
            raise ValueError("ExecuTorch detector input_shape must be [1, C, H, W]")

    def _prepare_input(self, image):
        if not isinstance(image, torch.Tensor):
            raise TypeError("ExecuTorch detector expects a CHW torch.Tensor image")

        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]

        if image.ndim != 3:
            raise ValueError(f"Expected CHW image, got {tuple(image.shape)}")

        original_height, original_width = image.shape[-2:]
        _, channels, height, width = self._input_shape

        if image.shape[0] != channels:
            raise ValueError(f"Model expects {channels} channels, got {image.shape[0]}")
        resized = functional.interpolate(image[None].to(torch.float32).cpu(), size=(height, width),
                                         mode="bilinear", align_corners=False)
        return resized, original_height, original_width

    def detect(self, image):
        tensor, original_height, original_width = self._prepare_input(image)
        _, _, input_height, input_width = tensor.shape
        outputs = self._method.execute((tensor,))
        keypoints, descriptors, scores = self._features(outputs, input_height, input_width)

        keypoints[:, 0] *= original_width / input_width
        keypoints[:, 1] *= original_height / input_height
        self._logger.info(f"Descriptor stats: mean={descriptors.mean():.4f}, std={descriptors.std():.4f}, "
                          f"norm_mean={descriptors.norm(dim=1).mean():.4f}")
        self._logger.info(f"ExecuTorch {self._detector_name} found {len(keypoints)} keypoints")
        return {"keypoints": keypoints, "descriptors": descriptors, "scores": scores,
                "width": original_width, "height": original_height, "executorch": True}


class ExecuTorchDescriptor(ExecuTorch, Descriptor, register=False):
    def __init__(self, descriptor_name, logger, config=None):
        if config is None:
            config = {}
        Descriptor.__init__(self, logger, descriptor_name)

        model_path = config.pop("executorch_model_path", config.pop("model_path", None))

        self._is_patch_based = self._descriptor_name not in ALL_DETECTORS

        if self._is_patch_based:
            if model_path is None:
                raise ValueError(f"ExecuTorch descriptor '{descriptor_name}' requires 'executorch_model_path'")

            self._method = self._load_method(model_path)
            self._patch_size = config.pop("patch_size", 32)

    @property
    def default_norm(self):
        return cv.NORM_L2

    def compute(self, image, features):
        if not self._is_patch_based:
            if features.get("executorch"):
                return features
            raise ValueError("ExecuTorch descriptor expects features created by ExecuTorchDetector")

        keypoints = features.get("kp")
        if not keypoints:
            self._logger.warning(f"{self._descriptor_name} received 0 keypoints; nothing to describe")
            return {"kp": keypoints, "des": ()}

        image = self._preprocess(image)
        patches = self._extract_patches(image, keypoints, self._patch_size).contiguous()

        descriptors = []
        for i in range(patches.shape[0]):
            single_patch = patches[i:i + 1].contiguous()
            output = self._method.execute((single_patch,))
            descriptors.append(output[0])
        descriptors = torch.cat(descriptors, dim=0)

        descriptors_np = descriptors.detach().cpu().numpy().astype(np.float32)
        self._logger.info(f"{self._descriptor_name} computed {descriptors_np.shape[0]} descriptors")
        return {"kp": keypoints, "des": descriptors_np}

    def _preprocess(self, image):
        if image.shape[2] == 3:
            img_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        else:
            img_rgb = image
        image = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        if not isinstance(image, torch.Tensor):
            raise TypeError("ExecuTorch descriptor expects a CHW/NCHW torch.Tensor image")
        if image.ndim == 3:
            image = image[None]

        image = image.to(torch.float32).cpu()
        return image

    def _extract_patches(self, image, keypoints, patch_size):
        if image.shape[1] == 3:
            image = image[:, 0:1] * 0.2989 + image[:, 1:2] * 0.5870 + image[:, 2:3] * 0.1140

        _, _, height, width = image.shape
        if hasattr(keypoints[0], 'pt'):
            xy = torch.tensor([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=torch.float32)
        else:
            xy = keypoints.to(torch.float32)

        n = xy.shape[0]
        half = patch_size / 2.0

        lin = torch.linspace(-half + 0.5, half - 0.5, patch_size)
        grid_y, grid_x = torch.meshgrid(lin, lin, indexing="ij")
        offsets = torch.stack((grid_x, grid_y), dim=-1)

        centers = xy.view(n, 1, 1, 2)
        sample_px = centers + offsets

        sample_px[..., 0] = sample_px[..., 0] / max(width - 1, 1) * 2 - 1
        sample_px[..., 1] = sample_px[..., 1] / max(height - 1, 1) * 2 - 1

        image_expanded = image.expand(n, -1, -1, -1)
        patches = functional.grid_sample(image_expanded, sample_px, align_corners=True)
        return patches


class ExecuTorchMatcher(ExecuTorch, Matcher, register=False):
    def __init__(self, logger, matcher_name, descriptor_name, config=None):
        if config is None:
            config = {}
        Matcher.__init__(self, logger, matcher_name, descriptor_name)

        model_path = config.get("executorch_model_path", config.get("model_path", None))
        if model_path is None:
            raise ValueError("ExecuTorch matcher requires 'executorch_model_path' in matcher config")

        self._method = self._load_method(model_path)
        self._num_keypoints = config.get("num_keypoints", 256)

    def _fixed_features(self, feature):
        keypoints, descriptors = feature["keypoints"], feature["descriptors"]
        if keypoints.shape[0] != self._num_keypoints:
            raise ValueError(f"Matcher expects K={self._num_keypoints}; detector returned {keypoints.shape[0]}")

        keypoints = keypoints[None].cpu().contiguous()
        descriptors = descriptors[None].cpu().contiguous()
        return {'keypoints': keypoints, 'descriptors': descriptors}

    def match(self, features0, features1):
        features0 = self._fixed_features(features0)
        features1 = self._fixed_features(features1)

        outputs = self._correspondences(features0, features1)
        matches0, _, match_scores0, _ = outputs
        valid = matches0[0] >= 0
        indices0 = torch.arange(self._num_keypoints)[valid]
        matches = torch.stack((indices0, matches0[0][valid].to(torch.long)), dim=1)
        return {"matches": matches, "scores": match_scores0[0][valid]}

    def _init_matcher(self):
        return None


class SuperPointLightGlueExecuTorch(ExecuTorchDetector, ExecuTorchDescriptor):
    def __init__(self, extractor_name, logger, config):
        if config is None:
            config = {}

        if "executorch_model_path" in config or "model_path" in config:
            ExecuTorchDetector.__init__(self, extractor_name, logger, config)
        else:
            ExecuTorchDescriptor.__init__(self, extractor_name, logger, config)

    def _features(self, outputs, input_height, input_width):
        _, probabilities, dense = outputs
        probabilities = probabilities[:, :-1]
        _, _, coarse_height, coarse_width = probabilities.shape

        scores = probabilities.permute(0, 2, 3, 1).reshape(1, coarse_height, coarse_width, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(1, coarse_height * 8, coarse_width * 8)[0]
        keypoints, values = self._nms_topk(scores, self._num_keypoints, self._nms_radius)
        descriptors = self._sample_descriptors(dense, keypoints / 8, coarse_height, coarse_width)
        return keypoints, descriptors, values


class DiskLightGlueExecuTorch(ExecuTorchDetector, ExecuTorchDescriptor):
    def __init__(self, extractor_name, logger, config):
        if config is None:
            config = {}

        if "executorch_model_path" in config or "model_path" in config:
            ExecuTorchDetector.__init__(self, extractor_name, logger, config)
        else:
            ExecuTorchDescriptor.__init__(self, extractor_name, logger, config)

    def _features(self, outputs, input_height, input_width):
        heatmap, dense = outputs
        keypoints, values = self._nms_topk(heatmap[0, 0], self._num_keypoints, self._nms_radius)
        descriptors = self._sample_descriptors(dense, keypoints, input_height, input_width)
        return keypoints, descriptors, values


class D2NetExecuTorch(ExecuTorchDetector, ExecuTorchDescriptor):
    def __init__(self, extractor_name, logger, config):
        if config is None:
            config = {}

        if "executorch_model_path" in config or "model_path" in config:
            ExecuTorchDetector.__init__(self, extractor_name, logger, config)
        else:
            ExecuTorchDescriptor.__init__(self, extractor_name, logger, config)

    def _d2net_score_map(self, dense: torch.Tensor):
        batch, channels, height, width = dense.shape
        exp_dense = torch.exp(dense - dense.amax(dim=(2, 3), keepdim=True))
        local_sum = functional.avg_pool2d(exp_dense, kernel_size=3, stride=1, padding=1) * 9
        alpha = exp_dense / (local_sum + 1e-8)

        channel_max = dense.amax(dim=1, keepdim=True)
        beta = dense / (channel_max + 1e-8)

        score_per_channel = alpha * beta
        score_map = score_per_channel.amax(dim=1)[0]
        return score_map

    def _features(self, outputs, input_height, input_width):
        dense = outputs[0]
        score_map = self._d2net_score_map(dense)
        coarse_keypoints, values = self._nms_topk(score_map, self._num_keypoints, self._nms_radius)

        coarse_height, coarse_width = score_map.shape
        keypoints = coarse_keypoints.clone()
        keypoints[:, 0] *= input_width / coarse_width
        keypoints[:, 1] *= input_height / coarse_height
        descriptors = self._sample_descriptors(dense, coarse_keypoints, coarse_height, coarse_width)
        return keypoints, descriptors, values


class TFeatExecuTorch(ExecuTorchDescriptor):
    pass


class HardNetExecuTorch(ExecuTorchDescriptor):
    pass


class LightGlueExecuTorch(ExecuTorchMatcher):
    def __init__(self, logger, matcher_name, descriptor_name, config=None):
        ExecuTorchMatcher.__init__(self, logger, matcher_name, descriptor_name, config)

    def _correspondences(self, features0, features1):
        keypoints0, descriptors0 = features0.get('keypoints', None), features0.get('descriptors', None)
        keypoints1, descriptors1 = features1.get('keypoints', None), features1.get('descriptors', None)
        outputs = self._method.execute((keypoints0, keypoints1, descriptors0, descriptors1))
        return outputs


class SuperGlueExecuTorch(ExecuTorchMatcher):
    def __init__(self, logger, matcher_name, descriptor_name, config=None):
        ExecuTorchMatcher.__init__(self, logger, matcher_name, descriptor_name, config)

    def _correspondences(self, features0, features1):
        keypoints0, descriptors0 = features0.get('keypoints', None), features0.get('descriptors', None)
        keypoints1, descriptors1 = features1.get('keypoints', None), features1.get('descriptors', None)
        scores0 = features0["scores"][None].cpu()
        scores1 = features1["scores"][None].cpu()
        outputs = self._method.execute((keypoints0, keypoints1, descriptors0.transpose(1, 2),
                                        descriptors1.transpose(1, 2), scores0, scores1))
        return outputs
