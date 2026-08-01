import torch
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
from pathlib import Path

from src.dnn_extractors import DNNFeatureExtractors
from src.image_utils import to_numpy_bgr


class SuperPoint(DNNFeatureExtractors):
    _image_processor = None

    def __init__(self, extractor_name, logger, config=None):
        if config is None:
            config = {}

        DNNFeatureExtractors.__init__(self, extractor_name, logger, config)
        checkpoint = config.pop('checkpoint', "weights/superpoint")
        local_files_only = config.pop('local_files_only', True)

        remote_repo = "magic-leap-community/superpoint"
        if config:
            self._logger.warning(f"SuperPoint: unknown config keys ignored: {list(config.keys())}")

        if SuperPoint._model is None:
            local_path = Path(checkpoint)
            if not local_path.exists() or not any(local_path.iterdir()):
                self._logger.warning(f"Local checkpoint {checkpoint} not found or empty.")
                self._logger.info(f"Switching to remote repository: {remote_repo}")
                checkpoint = remote_repo
                local_files_only = False

            try:
                self._logger.info(f"Loading SuperPoint from {checkpoint} (local={local_files_only})")
                SuperPoint._image_processor = AutoImageProcessor.from_pretrained(
                    checkpoint, local_files_only=local_files_only)
                SuperPoint._model = SuperPointForKeypointDetection.from_pretrained(
                    checkpoint, local_files_only=local_files_only).to(self._device)
            except Exception as e:
                self._logger.error(f"Failed to load from {checkpoint}: {e}")

        self._processor = SuperPoint._image_processor
        self._model = SuperPoint._model

    def _forward(self, img):
        if img is None:
            self._logger.error("Input image is None. Detection aborted.")
            return {'keypoints': (), 'descriptors': ()}

        self._logger.info(f"Running inference with {self._detector_name}")

        input_type = 'torch' if isinstance(img, torch.Tensor) else 'numpy'
        img = to_numpy_bgr(img, input_type=input_type)
        height, width = img.shape[:2]
        inputs = self._processor(img, return_tensors="pt").to(self._device)

        try:
            with torch.no_grad():
                outputs = self._model(**inputs)

            processed = self._processor.post_process_keypoint_detection(outputs, [img.shape[:2]])[0]

            raw_kp = processed['keypoints']
            raw_scores = processed['scores']
            raw_des = processed['descriptors']

            mask = raw_scores > self._threshold
            extracted = {
                'keypoints': raw_kp[mask],
                'descriptors': raw_des[mask],
                'scores': raw_scores[mask],
                'width': width,
                'height': height
            }
            SuperPoint._extracted_data = extracted
            if len(raw_kp[mask]) > 0:
                self._logger.info(f"{self._detector_name} found {len(raw_kp[mask])} points")
            else:
                self._logger.warning(f"{self._detector_name} found 0 points")

            if raw_des[mask] is not None:
                self._logger.info(f"{self._descriptor_name} computed {len(raw_des[mask])} descriptors")
            else:
                self._logger.warning(f"{self._descriptor_name} computed 0 descriptors")

            return extracted

        except Exception as e:
            self._logger.warning(f"{self._detector_name} inference failed (likely 0 points): {e}")
            return {'keypoints': (), 'descriptors': (), 'width': width, 'height': height}
