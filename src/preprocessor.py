import torch
from src.algorithms import DNN_ALGORITHMS, OPENCV_ALGORITHMS, DNN_PIPELINES
from src.converter import Converter


class Preprocessor:
    _VALID_FORMATS = {'tensor', 'opencv'}

    def __init__(self, logger, config=None):
        if config is None:
            config = {}

        self._device = config.get('device', 'cpu')
        self._logger = logger
        self._image_converter = Converter.create('image')
        self._features_converter = Converter.create('features')
        self._matches_converter = Converter.create('matches')

        self._features0 = None
        self._features1 = None

    def _get_format(self, algo):
        if algo in self._VALID_FORMATS:
            return algo

        if algo in DNN_ALGORITHMS:
            return 'tensor'

        if algo in OPENCV_ALGORITHMS:
            return 'opencv'

        raise ValueError(f"Unknown algorithm or format: '{algo}'. "
                         f"Expected one of {DNN_ALGORITHMS | OPENCV_ALGORITHMS | self._VALID_FORMATS}")

    def _log_conversion(self, data_type, from_algo, to_algo):
        from_format = self._get_format(from_algo)
        to_format = self._get_format(to_algo)

        if from_format != to_format:
            self._logger.info(f"Converting {data_type} from {from_format} to {to_format} on {self._device}")
        return from_format, to_format

    def prepare_image(self, img, from_algo, to_algo):
        from_format, to_format = self._log_conversion('image', from_algo, to_algo)
        return self._image_converter.convert(img, from_format=from_format,
                                             to_format=to_format, device=self._device)

    def prepare_features(self, features, from_algo, to_algo):
        if self._features0 is not None:
            features = self._features0
            self._features0 = None

        elif self._features1 is not None:
            features = self._features1
            self._features1 = None

        from_format, to_format = self._log_conversion('features', from_algo, to_algo)
        return self._features_converter.convert(features, from_format=from_format,
                                                to_format=to_format, device=self._device)

    def prepare_matches(self, correspondences, from_algo, to_algo):
        if from_algo in DNN_PIPELINES:
            kp0 = correspondences.get('keypoints0', torch.empty((0, 2)))
            kp1 = correspondences.get('keypoints1', torch.empty((0, 2)))

            self._features0 = {
                'keypoints': kp0,
                'descriptors': torch.empty((len(kp0), 0))
            }
            self._features1 = {
                'keypoints': kp1,
                'descriptors': torch.empty((len(kp1), 0))
            }

        from_format, to_format = self._log_conversion('matches', from_algo, to_algo)
        return self._matches_converter.convert(correspondences, from_format=from_format,
                                               to_format=to_format, device=self._device)
