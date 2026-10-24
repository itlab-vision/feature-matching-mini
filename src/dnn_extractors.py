import cv2 as cv
import numpy as np

from src.detectors import Detector
from src.descriptors import Descriptor
import src.backend.torch.model_wrapper  # noqa: E402, F401
import src.backend.torch.inference_api  # noqa: E402, F401
import src.backend.torch.io_adapter  # noqa: E402, F401
import src.backend.executorch.model_wrapper
import src.backend.executorch.io_adapter
import src.backend.executorch.inference_api
from src.backend.io_adapter_base import IOAdapter
from src.backend.inference_api_base import InferenceAPI
from src.backend.model_wrapper_base import ModelWrapper


class DNNFeatureExtractors(Detector, Descriptor):
    _model_wrapper = None
    _model_components = None
    _io_adapter = None
    _inference = None

    _is_extracted = False
    _extracted_data = {}

    def __init__(self, extractor_name, logger, config=None):
        if config is None:
            config = {}

        Detector.__init__(self, logger, extractor_name)
        Descriptor.__init__(self, logger, extractor_name)

        backend = config.pop('backend', 'torch').lower()
        self._nfeatures = config.get('nfeatures', 4096)
        self._threshold = config.get('threshold', 0.005)

        if DNNFeatureExtractors._model_wrapper is None:
            DNNFeatureExtractors._model_wrapper = ModelWrapper.create(backend=backend, model_name=extractor_name,
                                                                      config=config, logger=logger)
            DNNFeatureExtractors._model_components = DNNFeatureExtractors._model_wrapper.load()

        if DNNFeatureExtractors._io_adapter is None:
            DNNFeatureExtractors._io_adapter = IOAdapter.create(backend=backend, model_name=extractor_name,
                                                                config=config, logger=logger)

        if DNNFeatureExtractors._inference is None:
            DNNFeatureExtractors._inference = InferenceAPI.create(backend=backend, logger=logger, config=config,
                                                                  model_name=extractor_name,
                                                                  model_wrapper=DNNFeatureExtractors._model_wrapper)

    def _forward(self, img):
        if img is None:
            self._logger.error("Input image is None. Detection aborted.")
            return {'keypoints': (), 'descriptors': ()}

        self._logger.info(f"Running inference with {self._detector_name}")
        inputs = {'image': img, **DNNFeatureExtractors._model_components}

        inputs = DNNFeatureExtractors._io_adapter.preprocess(inputs)
        outputs = DNNFeatureExtractors._inference.run(inputs)
        outputs = DNNFeatureExtractors._io_adapter.postprocess(outputs)

        keypoints = outputs.pop('keypoints', ())
        descriptors = outputs.pop('descriptors', ())
        scores = outputs.pop('scores', ())

        mask = scores > self._threshold
        kp = keypoints[mask]
        des = descriptors[mask]
        sc = scores[mask]

        scales = outputs.get('scales')
        oris = outputs.get('oris')

        if scales is not None:
            outputs['scales'] = scales[:, mask]
        if oris is not None:
            outputs['oris'] = oris[:, mask]

        if self._nfeatures is not None and len(kp) > self._nfeatures:
            indices = np.argsort(sc)[::-1][:self._nfeatures]
            kp = kp[indices]
            des = des[indices]

            if scales is not None:
                outputs['scales'] = scales[:, mask]
            if oris is not None:
                outputs['oris'] = oris[:, mask]

        height, width = img.shape[:2]
        extracted = {'keypoints': kp, 'descriptors': des, 'scores': sc,
                     'height': height, 'width': width, **outputs}
        DNNFeatureExtractors._extracted_data = extracted

        if len(kp) > 0:
            self._logger.info(f"{self._detector_name} found {len(kp)} points")
        else:
            self._logger.warning(f"{self._detector_name} found 0 points")

        if des is not None:
            self._logger.info(f"{self._descriptor_name} computed {len(des)} descriptors")
        else:
            self._logger.warning(f"{self._descriptor_name} computed 0 descriptors")

        return extracted

    @property
    def default_norm(self):
        return cv.NORM_L2

    def detect(self, img):
        DNNFeatureExtractors._is_extracted = True
        return self._forward(img)

    def compute(self, img, features=None):
        if DNNFeatureExtractors._is_extracted:
            DNNFeatureExtractors._is_extracted = False
            return self._extracted_data
        else:
            return self._forward(img)

    def detectAndCompute(self, img):
        return self._forward(img)

    def close(self):
        DNNFeatureExtractors._inference = None
        DNNFeatureExtractors._io_adapter = None
        DNNFeatureExtractors._model_components = None
        DNNFeatureExtractors._model_wrapper = None
        DNNFeatureExtractors._extracted_data = {}
