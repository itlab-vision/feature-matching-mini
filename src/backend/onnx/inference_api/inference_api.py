from src.backend.inference_api_base import InferenceAPI
import cv2 as cv


@InferenceAPI.register("onnx")
@InferenceAPI.register("aliked_onnx")
@InferenceAPI.register("disk_onnx")
@InferenceAPI.register("lightglue_onnx")
class OnnxInferenceAPI(InferenceAPI):
    _OUTPUT_NAMES = {
        'aliked': ["keypoints", "descriptors", "scores"],
        'disk': ["keypoints", "descriptors", "scores"],
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
            if self._model_name in ('disk', 'aliked'):
                size = (1024, 1024) if self._model_name == 'disk' else (640, 640)
                blob = cv.dnn.blobFromImage(inputs['image'], scalefactor=1.0 / 255.0,
                                            size=size, swapRB=True, crop=False)
                inputs = {'image': blob}
            if self._model_name == 'lightglue':
                inputs = {name: inputs[f'features{index}'][field][None]
                          for index in (0, 1)
                          for name, field in ((f'kpts{index}', 'normalized_keypoints'),
                                              (f'desc{index}', 'descriptors'))}
            outputs = self._model_wrapper.run(self.output_names, inputs)
            if self._model_name == 'lightglue' and outputs is not None:
                matches, scores = outputs
                matches = matches[0] if matches.ndim == 3 else matches
                scores = scores[0] if scores.ndim == 2 else scores
                valid = scores > self._config.get('score_threshold', 0.1)
                return {'matches': matches[valid], 'scores': scores[valid]}
            return outputs

        except Exception as e:
            self._logger.error(f"{self._model_name} inference error: {e}")
