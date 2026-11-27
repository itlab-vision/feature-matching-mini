from src.backend.inference_api_base import InferenceAPI
import cv2 as cv
import numpy as np

from src.backend.lightglue_utils import lightglue_inputs, lightglue_matches


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
            if self._model_name in ('disk', 'disk_lightglue', 'aliked', 'aliked_lightglue', 'superpoint_lightglue'):
                default_size = {'disk': (1024, 1024), 'aliked': (640, 640), 'superpoint': (480, 640)}
                height, width = self._config.get(
                    'input_shape', default_size[self._model_name.removesuffix('_lightglue')]
                )[-2:]
                blob = cv.dnn.blobFromImage(inputs['image'], scalefactor=1.0 / 255.0,
                                            size=(width, height), swapRB=True, crop=False)
                inputs = {'image': blob}
            if self._model_name == 'lightglue':
                if inputs is None:
                    return None
                inputs = lightglue_inputs(inputs)
                counts = tuple(inputs[f'kpts{i}'].shape[1] for i in range(2))
                for model_input in self._model_wrapper.get_inputs():
                    count = model_input.shape[1]
                    if isinstance(count, int):
                        data = inputs[model_input.name][:, :count]
                        padding = [(0, 0), (0, max(0, count - data.shape[1]))]
                        inputs[model_input.name] = np.pad(data, padding + [(0, 0)] * (data.ndim - 2))
            outputs = self._model_wrapper.run(self.output_names, inputs)
            if self._model_name == 'lightglue' and outputs is not None:
                result = lightglue_matches(outputs, self._config.get('score_threshold', 0.1))
                valid = (result['matches'] < counts).all(axis=1)
                return {key: value[valid] for key, value in result.items()}
            return outputs

        except Exception as e:
            self._logger.error(f"{self._model_name} inference error: {e}")
