from pathlib import Path
from executorch.runtime import Runtime, Verification

from src.backend.model_wrapper_base import ModelWrapper


@ModelWrapper.register("executorch")
class ExecuTorchModelWrapper(ModelWrapper):
    def __init__(self, model_name, model_path=None, config=None, logger=None):
        if config is None:
            config = {}

        super().__init__(model_name, model_path, config, logger)
        self._program = None
        self._method = None

    def load(self):
        model_path = self._model_path or self._config.get("executorch_model_path",
                                                          self._config.get("model_path"))
        if model_path is None:
            raise ValueError("ExecuTorch requires 'model_path' in config")

        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"ExecuTorch model does not exist: {model_path}")

        self._program = Runtime.get().load_program(str(model_path), verification=Verification.Minimal)
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
