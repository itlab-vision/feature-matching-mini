from abc import abstractmethod
import numpy as np

from src.dnn_extractors import DNNFeatureExtractors
from src.dnn_matchers import DNNMatcher
import src.backend.torch.model_wrapper
import src.backend.torch.inference_api
import src.backend.torch.io_adapter
from src.backend.io_adapter_base import IOAdapter
from src.backend.inference_api_base import InferenceAPI
from src.backend.model_wrapper_base import ModelWrapper


class DNNPipeline(DNNFeatureExtractors, DNNMatcher):
    _model_wrapper = None
    _model_components = None
    _io_adapter = None
    _inference = None

    def __init__(self, extractor_name, logger, config=None):
        if config is None:
            config = {}

        DNNFeatureExtractors.__init__(self, extractor_name, logger, config)
        # DNNMatcher.__init__(self, extractor_name, logger, config, extractor_name)
        #
        # backend = config.pop('backend', 'torch').lower()
        # self._nfeatures = config.get('nfeatures', 4096)
        # self._threshold = config.get('threshold', 0.005)
        #
        # if DNNPipeline._model_wrapper is None:
        #     DNNPipeline._model_wrapper = ModelWrapper.create(backend=backend, model_name=extractor_name,
        #                                                               config=config, logger=logger)
        #     DNNPipeline._model_components = DNNPipeline._model_wrapper.load()
        #
        # if DNNPipeline._io_adapter is None:
        #     DNNPipeline._io_adapter = IOAdapter.create(backend=backend, model_name=extractor_name,
        #                                                         config=config, logger=logger)
        #
        # if DNNPipeline._inference is None:
        #     DNNPipeline._inference = InferenceAPI.create(backend=backend, logger=logger, config=config,
        #                                                           model_name=extractor_name,
        #                                                           model_wrapper=DNNPipeline._model_wrapper)

    def detect(self, img):
        return {'image': img, 'keypoints': (), 'descriptors': ()}

    def compute(self, img, features=None):
        return features

    def match(self, features0, features1):
        img0 = features0.get('image')
        img1 = features1.get('image')

        if img0 is None or img1 is None:
            self._logger.error("Input image is None")
            return {'matches': (), 'scores': ()}

        inputs = {'image0': img0, 'image1': img1}
        inputs = DNNFeatureExtractors._io_adapter.preprocess(inputs)
        outputs = DNNFeatureExtractors._inference.run(inputs)
        outputs = DNNFeatureExtractors._io_adapter.postprocess(outputs)

        matches = outputs.get('matches')
        scores = outputs.get('scores')
        keypoints0 = outputs.get('keypoints0')
        keypoints1 = outputs.get('keypoints1')

        if matches is None or scores is None:
            self._logger.warning(f"{self._detector_name}: invalid matcher output")
            return {'matches': (), 'scores': ()}

        mask = scores > self._threshold
        matches = matches[mask]
        scores = scores[mask]

        if self._nfeatures is not None and len(matches) > self._nfeatures:
            indices = np.argsort(scores)[::-1][:self._nfeatures]
            matches = matches[indices]
            scores = scores[indices]

        extracted = {'keypoints0': keypoints0,
                     'keypoints1': keypoints1,
                     'matches': matches,
                     'scores': scores,
                     **outputs}
        DNNPipeline._extracted_data = extracted

        if len(matches) > 0:
            self._logger.info(f"{self._detector_name} found {len(matches)} matches")
        else:
            self._logger.warning(f"{self._detector_name} found 0 matches")

        return extracted

    def close(self):
        DNNFeatureExtractors._inference = None
        DNNFeatureExtractors._io_adapter = None
        DNNFeatureExtractors._model_components = None
        DNNFeatureExtractors._model_wrapper = None
        DNNFeatureExtractors._extracted_data = {}
