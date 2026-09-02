import torch
import sys
from pathlib import Path

from src.dnn_extractors import DNNFeatureExtractors  # noqa: E402

LIGHTGLUE_ROOT = Path(__file__).resolve().parent.parent / "3rdparty" / "lightglue"
if str(LIGHTGLUE_ROOT) not in sys.path:
    sys.path.insert(0, str(LIGHTGLUE_ROOT))

from lightglue import SuperPoint, DISK, SIFT, ALIKED, DoGHardNet  # noqa: E402


class LightGlueFeatureExtractor(DNNFeatureExtractors, register=False):
    _EXTRACTOR_CLASSES = {
        'superpoint_lightglue': SuperPoint,
        'disk_lightglue': DISK,
        'sift_lightglue': SIFT,
        'aliked_lightglue': ALIKED,
        'doghardnet_lightglue': DoGHardNet
    }

    _shared_models = {}

    def __init__(self, extractor_name, logger, config=None):
        if config is None:
            config = {}

        DNNFeatureExtractors.__init__(self, extractor_name, logger, config)

        model_key = (self._detector_name, self._device.type)
        if model_key not in LightGlueFeatureExtractor._shared_models:
            extractor_class = self._EXTRACTOR_CLASSES.get(self._detector_name)
            if not extractor_class:
                raise ValueError(f"Extractor '{extractor_name}' not found.")

            self._logger.info(f"Loading {self._detector_name} weights onto {self._device}")
            LightGlueFeatureExtractor._shared_models[model_key] = (extractor_class(**config).eval().to(self._device))

        self._extractor = LightGlueFeatureExtractor._shared_models[model_key]

    def _forward(self, image_tensor):
        if image_tensor is None:
            self._logger.error("Input image tensor is None.")
            return {'keypoints': (), 'descriptors': ()}

        with torch.no_grad():
            if image_tensor.ndim == 3:
                image_tensor = image_tensor[None]

            self._logger.info(f"Running inference with {self._detector_name}")
            extracted = self._extractor.extract(image_tensor.to(self._device))

            num_found = extracted['keypoints'].shape[1]
            if self._nfeatures is not None and num_found > self._nfeatures:
                scores = extracted.get('keypoint_scores')
                if scores is not None:
                    _, indices = torch.topk(scores, k=self._nfeatures, dim=1)

                    extracted['keypoints'] = torch.take_along_dim(
                        extracted['keypoints'], indices.unsqueeze(-1), dim=1)
                    extracted['descriptors'] = torch.take_along_dim(
                        extracted['descriptors'], indices.unsqueeze(-1), dim=1)

                    if 'keypoint_scores' in extracted:
                        extracted['keypoint_scores'] = torch.take_along_dim(
                            extracted['keypoint_scores'], indices, dim=1)
                else:
                    extracted['keypoints'] = extracted['keypoints'][:, :self._nfeatures]
                    extracted['descriptors'] = extracted['descriptors'][:, :self._nfeatures]

            LightGlueFeatureExtractor._extracted_data = extracted
            self._logger.info(f"{self._detector_name} found"
                              f" {LightGlueFeatureExtractor._extracted_data['keypoints'].shape[1]} points")
            return extracted
