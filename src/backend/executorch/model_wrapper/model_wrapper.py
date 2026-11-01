from pathlib import Path

try:
    from executorch.runtime import Runtime, Verification
except ModuleNotFoundError:
    Runtime = None
    Verification = None

from src.backend.model_wrapper_base import ModelWrapper


EXECUTORCH_DETECTOR_MODELS = {
    'd2net': 'd2net_xnnpack_1x3x480x640.pte',
    'disk_lightglue': 'disk_lightglue_xnnpack_1x3x480x640.pte',
    'superpoint_lightglue': 'superpoint_lightglue_xnnpack_1x3x480x640.pte',
    'xfeat': 'xfeat_xnnpack_1x3x480x640.pte',
}
EXECUTORCH_DESCRIPTOR_MODELS = {
    'd2net': 'd2net_xnnpack_1x3x480x640.pte',
    'disk_lightglue': 'disk_lightglue_xnnpack_1x3x480x640.pte',
    'superpoint_lightglue': 'superpoint_lightglue_xnnpack_1x3x480x640.pte',
    'xfeat': 'xfeat_xnnpack_1x3x480x640.pte',
    'hardnet': 'hardnet_xnnpack_1x1x32x32.pte',
    'tfeat': 'tfeat_xnnpack_1x1x32x32.pte',
}
EXECUTORCH_MATCHER_MODELS = {
    'disk_lightglue': 'lightglue_disk_k256_xnnpack_1x3x480x640.pte',
    'superpoint_lightglue': 'lightglue_superpoint_k256_xnnpack_1x3x480x640.pte',
}


@ModelWrapper.register("executorch")
class ExecuTorchModelWrapper(ModelWrapper):
    def __init__(self, model_name, model_path=None, config=None, logger=None):
        if config is None:
            config = {}

        super().__init__(model_name, model_path, config, logger)
        self._program = None
        self._method = None

    def load(self):
        model_path = self._model_path or self._config.get("executorch_model_path", self._config.get("model_path"))
        if model_path is None:
            raise ValueError("ExecuTorch requires 'model_path' in config")

        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"ExecuTorch model does not exist: {model_path}")

        if Runtime is None:
            raise ModuleNotFoundError(
                "ExecuTorch runtime is not installed. Install 'executorch' to load .pte models."
            )

        verification = Verification.Minimal if Verification is not None else None
        self._program = Runtime.get().load_program(str(model_path), verification=verification)
        self._method = self._program.load_method(self._config.get("method_name", "forward"))
        self._model = self._method

        if self._logger:
            self._logger.info(f"Loaded ExecuTorch model '{self._model_name}' from {model_path}")

        return {"model": self._model}

    def call(self, inputs):
        if self._method is None:
            raise RuntimeError("ExecuTorch model is not loaded. Call load() first.")

        if not isinstance(inputs, (tuple, list)):
            inputs = (inputs,)

        return self._method.execute(tuple(inputs))
