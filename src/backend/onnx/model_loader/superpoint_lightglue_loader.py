import onnxruntime as ort

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.onnx.model_loader.model_loader import OnnxModelLoader
from src.backend.onnx.onnx_utils import OnnxSessionConfiguration


@ModelWrapper.register("superpoint_lightglue_onnx")
class SuperPointLightGlueOnnxModelLoader(OnnxModelLoader):
    def load(self):
        model_path = self._config.get("superpoint_model_path")
        if model_path is None:
            raise ValueError("SuperPoint ONNX requires 'superpoint_model_path'")
        self._logger.info(f"Initializing SuperPoint from {model_path} (ONNX Runtime)")
        session_config = OnnxSessionConfiguration(self._logger, self._config)
        session = ort.InferenceSession(
            model_path,
            sess_options=session_config.set_session_options(),
            providers=session_config.get_providers_with_options(),
        )
        self._model = session
        return {"model": self._model}
