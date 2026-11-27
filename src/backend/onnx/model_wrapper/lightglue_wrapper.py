import onnxruntime as ort
from src.backend.model_wrapper_base import ModelWrapper
from src.backend.onnx.model_wrapper.model_wrapper import OnnxModelWrapper
from src.backend.onnx.utils import OnnxSessionConfiguration


@ModelWrapper.register("lightglue_onnx")
class LightGlueOnnxModelWrapper(OnnxModelWrapper):
    def load(self):
        descriptor = self._config.get('descriptor_class')
        feature = descriptor._descriptor_name.removesuffix('_lightglue') if descriptor is not None else 'disk'
        model_path = self._config.pop('lightglue_model_path', None) or f'models/lightglue_{feature}_k512.onnx'
        self._logger.info(f"Initializing {self._model_name} from {model_path} (ONNX Runtime)")
        try:
            session_config = OnnxSessionConfiguration(self._logger, self._config)
            sess_options = session_config.set_session_options()
            providers = session_config.get_providers_with_options()

            session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=providers
            )
            self._model = session
            return {'model': self._model}

        except Exception as e:
            self._logger.error(f"Failed to load ONNX model {self._model_name} from {model_path}: {e}")
            raise
