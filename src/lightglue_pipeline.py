import torch

from src.dnn_extractors import DNNFeatureExtractors
from lightglue import SuperPoint, DISK, SIFT, ALIKED, DoGHardNet


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
            LightGlueFeatureExtractor._extracted_data = extracted
            self._logger.info(f"{self._detector_name} found"
                              f" {LightGlueFeatureExtractor._extracted_data['keypoints'].shape[1]} points")
            return extracted
