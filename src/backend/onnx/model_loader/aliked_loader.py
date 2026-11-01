import onnxruntime as ort
from src.backend.model_wrapper_base import ModelWrapper
from src.backend.onnx.model_loader.model_loader import OnnxModelLoader
from src.backend.onnx.onnx_utils import OnnxSessionConfiguration


@ModelWrapper.register("aliked_onnx")
class AlikedOnnxModelLoader(OnnxModelLoader):
    def load(self):
        model_path = self._config.pop('aliked_model_path', "models/aliked-n32-top2k-640.onnx")
        self._logger.info(f"Initializing {self._model_name} from {model_path} (ONNX Runtime)")
        try:
            session_config = OnnxSessionConfiguration(self._logger, self._config)
            sess_options = session_config.set_session_options()
            providers = session_config.get_providers_with_options()

            session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers
            )
            self._model = session
            return {'model': self._model}

        except Exception as e:
            self._logger.error(f"Failed to load ONNX model {self._model_name} from {model_path}: {e}")
            raise
