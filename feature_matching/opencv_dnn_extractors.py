import cv2 as cv

from feature_matching.descriptors import Descriptor
from feature_matching.detectors import Detector


class OpenCVDNNFeatureExtractors(Detector, Descriptor, register=False):
    _is_extracted = False
    _extracted_data = {}

    def __init__(self, extractor_name, logger, extractor):
        Detector.__init__(self, logger, extractor_name)
        Descriptor.__init__(self, logger, extractor_name)
        self.extractor_name = extractor_name
        self.extractor = extractor

    @property
    def default_norm(self):
        return cv.NORM_L2

    def _forward(self, img):
        if img is None:
            self._logger.error("Input image is None. Detection aborted.")
            return {'kp': (), 'des': ()}

        self._logger.info(f"Running inference with {self._detector_name}")

        kp, des = self.extractor.detectAndCompute(img, None)
        self._logger.info(f"{self.extractor_name} found {len(kp)} points")
        OpenCVDNNFeatureExtractors._extracted_data = {'kp': kp, 'des': des, 'img_shape': img.shape}
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
    _PARAM_MAPPING = {
        'nfeatures': 'max_num_keypoints',
        'threshold': 'detection_threshold',
        'scale_factor': 'nms_radius'
    }

    def __init__(self, extractor_name, logger, config):
        aliked_model_path = config.pop('aliked_model_path', "models/aliked-n32-top2k-640.onnx")
        mapped_config = {}
        for config_key, value in config.items():
            if config_key in self._PARAM_MAPPING:
                mapped_config[self._PARAM_MAPPING[config_key]] = value

        aliked_params = cv.ALIKED.Params()
        for key, value in mapped_config.items():
            if hasattr(aliked_params, key):
                setattr(aliked_params, key, value)
        super().__init__(extractor_name, logger, cv.ALIKED.create(aliked_model_path, aliked_params))


class DISKOpenCV(OpenCVDNNFeatureExtractors):
    _PARAM_MAPPING = {
        'nfeatures': 'maxKeypoints',
        'threshold': 'scoreThreshold'
    }

    def __init__(self, extractor_name, logger, config):
        disk_model_path = config.pop('disk_model_path', "models/disk_1024.onnx")
        mapped_config = {}
        for config_key, value in config.items():
            if config_key in self._PARAM_MAPPING:
                mapped_config[self._PARAM_MAPPING[config_key]] = value
        super().__init__(extractor_name, logger, cv.DISK.create(disk_model_path, **mapped_config))
