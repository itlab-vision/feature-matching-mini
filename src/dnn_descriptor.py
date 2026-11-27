import cv2 as cv
import numpy as np

from src.descriptors import Descriptor
from src.backend.io_adapter_base import IOAdapter
from src.backend.inference_api_base import InferenceAPI
from src.backend.model_wrapper_base import ModelWrapper


class DNNDescriptor(Descriptor):
    _model_wrapper = None
    _model_components = None
    _io_adapter = None
    _inference = None

    def __init__(self, descriptor_name, logger, config=None):
        if config is None:
            config = {}
        else:
            config = config.copy()

        Descriptor.__init__(self, logger, descriptor_name)

        backend = config.pop('backend', 'torch').lower()
        self._backend = backend

        if DNNDescriptor._model_wrapper is None:
            DNNDescriptor._model_wrapper = ModelWrapper.create(backend=backend, model_name=descriptor_name,
                                                               config=config, logger=logger)

        if DNNDescriptor._io_adapter is None:
            DNNDescriptor._model_components = self._model_wrapper.load()
            DNNDescriptor._io_adapter = IOAdapter.create(backend=backend, model_name=descriptor_name,
                                                         model_wrapper=self._model_wrapper, config=config,
                                                         logger=logger)

        if DNNDescriptor._inference is None:
            DNNDescriptor._inference = InferenceAPI.create(backend=backend, logger=logger, config=config,
                                                           model_name=descriptor_name,
                                                           model_wrapper=self._model_wrapper)

    @property
    def default_norm(self):
        return cv.NORM_L2

    def compute(self, img, features):
        if img is None:
            self._logger.error('Input image is None. Descriptor computation aborted.')
            return {'keypoints': np.empty(0, dtype=object),
                    'descriptors': np.empty((0, 128), dtype=np.float32)}

        self._logger.info(f'Running inference with {self._descriptor_name} on {self._backend} backend')
        keypoints = features.get('keypoints', features.get('kp'))
        if keypoints is None:
            keypoints = np.empty(0, dtype=object)
        inputs = {'image': img, 'keypoints': keypoints, 'scores': features.get('scores'),
                  **DNNDescriptor._model_components}
        inputs = DNNDescriptor._io_adapter.preprocess(inputs)
        outputs = DNNDescriptor._inference.run(inputs)
        outputs = DNNDescriptor._io_adapter.postprocess(outputs)

        self._logger.info(f'{self._descriptor_name} computed {len(outputs["descriptors"])} descriptors')
        return outputs

    def close(self):
        DNNDescriptor._inference = None
        DNNDescriptor._io_adapter = None
        DNNDescriptor._model_components = None
        DNNDescriptor._model_wrapper = None
