from pathlib import Path
import torch
from transformers import AutoImageProcessor, SuperPointForKeypointDetection

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.torch.model_wrapper.model_wrapper import TorchModelWrapper


@ModelWrapper.register("superpoint_torch")
class SuperPointTorchModelWrapper(TorchModelWrapper):
    def load(self):
        checkpoint = self._config.pop('checkpoint', "weights/superpoint")
        local_files_only = self._config.pop('local_files_only', True)
        remote_repo = "magic-leap-community/superpoint"

        local_path = Path(checkpoint)
        if not local_path.exists() or not any(local_path.iterdir()):
            self._logger.warning(f"Local checkpoint {checkpoint} not found or empty.")
            self._logger.info(f"Switching to remote repository: {remote_repo}")
            checkpoint = remote_repo
            local_files_only = False

        try:
            self._logger.info(f"Loading SuperPoint from {checkpoint} (local={local_files_only})")
            self._image_processor = AutoImageProcessor.from_pretrained(
                checkpoint, local_files_only=local_files_only)
            self._model = SuperPointForKeypointDetection.from_pretrained(
                checkpoint, local_files_only=local_files_only).to(self._device)
            return {'model': self._model, 'processor': self._image_processor}

        except Exception as e:
            self._logger.error(f"Failed to load from {checkpoint}: {e}")

    def call(self, inputs):
        img = inputs.get("image")

        try:
            with torch.no_grad():
                outputs = self._model(**img)

            return {'inference_outputs': outputs, **inputs}

        except Exception as e:
            self._logger.error(f"Super Point inference error: {e}")
            raise
