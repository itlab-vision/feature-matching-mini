from src.backend.inference_api_base import InferenceAPI


@InferenceAPI.register("onnx")
@InferenceAPI.register("aliked_onnx")
@InferenceAPI.register("disk_onnx")
@InferenceAPI.register("lightglue_onnx")
class OnnxInferenceAPI(InferenceAPI):
    _OUTPUT_NAMES = {
        'aliked': ["keypoints", "descriptors", "scores"],
        'aliked_lightglue': ["keypoints", "descriptors", "scores"],
        'disk': ["keypoints", "descriptors", "scores"],
        'disk_lightglue': ["keypoints", "descriptors", "scores"],
        'superpoint_lightglue': ["score_logits", "probabilities", "descriptors"],
        'lightglue': ["matches0", "mscores0"],
    }

    def __init__(self, logger, model_name, model, config=None):
        if config is None:
            config = {}
        super().__init__(logger, model_name, model, config)

        self._model_name = model_name
        self.output_names = self._OUTPUT_NAMES.get(model_name)

    def run(self, inputs=None):
        try:
            outputs = self._model_wrapper.run(self.output_names, inputs)
            return outputs

        except Exception as e:
            self._logger.error(f"{self._model_name} inference error: {e}")
