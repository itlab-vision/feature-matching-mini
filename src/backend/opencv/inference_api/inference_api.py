from src.backend.inference_api_base import InferenceAPI


@InferenceAPI.register("opencv")
@InferenceAPI.register("aliked_opencv")
@InferenceAPI.register("disk_opencv")
@InferenceAPI.register("lightglue_opencv")
class OpenCVInferenceAPI(InferenceAPI):
    _OUTPUT_NAMES = {
        'aliked_opencv': ["keypoints", "descriptors", "scores"],
        'disk_opencv': ["keypoints", "descriptors", "scores"],
        'lightglue_opencv': ["matches0", "mscores0"]
    }

    def __init__(self, logger, model_name, model, config=None):
        if config is None:
            config = {}
        super().__init__(logger, model_name, model, config)

        self._model_name = model_name
        self.output_names = self._OUTPUT_NAMES.get(model_name)

    def run(self, inputs=None):
        try:
            outputs = self._model_wrapper.forward(self.output_names)
            return outputs

        except Exception as e:
            self._logger.error(f"{self._model_name} inference error: {e}")
