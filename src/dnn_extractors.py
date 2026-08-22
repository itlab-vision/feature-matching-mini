import torch
import cv2 as cv

from src.detectors import Detector
from src.descriptors import Descriptor


class DNNFeatureExtractors(Detector, Descriptor, register=False):
    _model = None
    _is_extracted = False
    _extracted_data = {}

    def __init__(self, extractor_name, logger, config=None):
        if config is None:
            config = {}

        Detector.__init__(self, logger, extractor_name)
        Descriptor.__init__(self, logger, extractor_name)

        device = config.get('device', None)
        self._threshold = config.get('threshold', 0.005)

        if device is None:
            if torch.cuda.is_available():
                self._device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self._device = torch.device('mps')
            else:
                self._device = torch.device('cpu')
        else:
            self._device = torch.device(device)

    @property
    def default_norm(self):
        return cv.NORM_L2

    def _forward(self, img):
        pass

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
