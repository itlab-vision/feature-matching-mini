from src.backend.inference_api_base import InferenceAPI
import cv2 as cv

from src.backend.lightglue_utils import lightglue_inputs, lightglue_matches


@InferenceAPI.register("opencv")
@InferenceAPI.register("aliked_opencv")
@InferenceAPI.register("disk_opencv")
@InferenceAPI.register("lightglue_opencv")
class OpenCVInferenceAPI(InferenceAPI):
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
            if self._model_name in ('disk', 'disk_lightglue', 'aliked', 'aliked_lightglue', 'superpoint_lightglue'):
                default_size = {'disk': (1024, 1024), 'aliked': (640, 640), 'superpoint': (480, 640)}
                height, width = self._config.get(
                    'input_shape', default_size[self._model_name.removesuffix('_lightglue')]
                )[-2:]
                blob = cv.dnn.blobFromImage(inputs['image'], scalefactor=1.0 / 255.0,
                                            size=(width, height), swapRB=True, crop=False)
                self._model_wrapper.setInput(blob)
            if self._model_name == 'lightglue':
                if inputs is None:
                    return None
                for name, value in lightglue_inputs(inputs).items():
                    self._model_wrapper.setInput(value, name=name)
            outputs = self._model_wrapper.forward(self.output_names)
            if self._model_name == 'lightglue' and outputs is not None:
                return lightglue_matches(outputs, self._config.get('score_threshold', 0.1))
            return outputs

        except Exception as e:
            self._logger.error(f"{self._model_name} inference error: {e}")
