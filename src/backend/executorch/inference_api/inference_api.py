from src.backend.inference_api_base import InferenceAPI


@InferenceAPI.register("executorch")
class ExecuTorchInferenceAPI(InferenceAPI):
    def __init__(self, logger, model_name, model_wrapper, config=None):
        if config is None:
            config = {}

        super().__init__(logger, model_name, model_wrapper, config)

    def run(self, inputs):
        return self._model_wrapper.call(inputs)
