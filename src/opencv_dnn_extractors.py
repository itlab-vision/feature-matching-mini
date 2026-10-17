import cv2 as cv
import numpy as np

from src.descriptors import Descriptor
from src.detectors import Detector

from src.backend.io_adapter_base import IOAdapter
from src.backend.inference_api_base import InferenceAPI
from src.backend.model_wrapper_base import ModelWrapper


class OpenCVDNNFeatureExtractors(Detector, Descriptor, register=False):
    _is_extracted = False
    _extracted_data = {}

    def __init__(self, extractor_name, logger, config):
        Detector.__init__(self, logger, extractor_name)
        Descriptor.__init__(self, logger, extractor_name)
        self.extractor_name = extractor_name

        self._loader = ModelWrapper.create(backend="opencv", model_name=extractor_name, config=config, logger=logger)
        self._model = self._loader.load()
        self._io_adapter = IOAdapter.create(backend="opencv", model_name=extractor_name,
                                            model_wrapper=self._model, config=config, logger=logger)
        self._inference = InferenceAPI.create(backend="opencv", logger=logger, model_name=extractor_name,
                                              model_wrapper=self._model, config=config)

        self._nfeatures = config.get('nfeatures', 4096)
        self._threshold = config.get('threshold', 0.005)

    @property
    def default_norm(self):
        return cv.NORM_L2

    def _forward(self, img):
        if img is None:
            self._logger.error("Input image is None. Detection aborted.")
            return {'keypoints': (), 'descriptors': ()}

        self._logger.info(f"Running inference with {self._detector_name}")

        self._io_adapter.preprocess({'image': img})
        outputs = self._inference.run()
        outputs = self._io_adapter.postprocess(outputs)

        kp = outputs.get('keypoints', np.array([]))
        des = outputs.get('descriptors', np.array([]))
        sc = outputs.get('scores', np.array([]))

        mask = sc > self._threshold
        kp = kp[mask]
        des = des[mask]
        sc = sc[mask]

        if self._nfeatures is not None and len(kp) > self._nfeatures:
            indices = np.argsort(sc)[::-1][:self._nfeatures]
            kp = kp[indices]
            des = des[indices]

        if len(kp) > 0:
            self._logger.info(f"{self.extractor_name} found {len(kp)} points")
        else:
            self._logger.warning(f"{self.extractor_name} found 0 points")

        OpenCVDNNFeatureExtractors._extracted_data = {'keypoints': kp, 'descriptors': des, 'scores': sc,
                                                      'img_shape': img.shape, 'extractor_name': self.extractor_name}
        return OpenCVDNNFeatureExtractors._extracted_data

    def detect(self, img):
        OpenCVDNNFeatureExtractors._is_extracted = True
        return self._forward(img)

    def compute(self, img, features):
        if OpenCVDNNFeatureExtractors._is_extracted:
            OpenCVDNNFeatureExtractors._is_extracted = False
            return OpenCVDNNFeatureExtractors._extracted_data
        else:
            return self._forward(img)

    def detectAndCompute(self, img):
        return self._forward(img)


class ALIKEDOpenCV(OpenCVDNNFeatureExtractors):
    def __init__(self, extractor_name, logger, config):
        super().__init__(extractor_name, logger, config)


class DISKOpenCV(OpenCVDNNFeatureExtractors):
    def __init__(self, extractor_name, logger, config):
        super().__init__(extractor_name, logger, config)
