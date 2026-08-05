from abc import ABC, abstractmethod
import cv2 as cv
import numpy as np
import torch


class Converter(ABC):
    _CONVERTERS = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        key = cls.__name__.replace("Converter", "").lower()
        if key:
            Converter._CONVERTERS[key] = cls

    @staticmethod
    def create(converter_type):
        if converter_type not in Converter._CONVERTERS:
            raise ValueError(f"Converter '{converter_type}' not found."
                             f" Available: {list(Converter._CONVERTERS.keys())}")
        return Converter._CONVERTERS[converter_type]()

    def convert(self, data, from_format, to_format, device='cpu'):
        if from_format == to_format:
            return data

        if from_format == 'opencv' and to_format == 'tensor':
            return self._to_tensor(data, device)

        if from_format == 'tensor' and to_format == 'opencv':
            return self._to_cv(data)

        return data

    @abstractmethod
    def _to_cv(self, data):
        pass

    @abstractmethod
    def _to_tensor(self, data, device='cpu'):
        pass


class ImageConverter(Converter):
    def _to_tensor(self, data, device='cpu'):
        if data.shape[2] == 3:
            img_rgb = cv.cvtColor(data, cv.COLOR_BGR2RGB)
        else:
            img_rgb = data
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        return tensor.to(device)

    def _to_cv(self, data):
        img_numpy = data.detach().cpu().numpy()
        if img_numpy.ndim == 2:
            img_numpy = img_numpy[:, :, np.newaxis]
        if img_numpy.shape[0] in [1, 3]:
            img_numpy = np.transpose(img_numpy, (1, 2, 0))
        if img_numpy.max() <= 1.0:
            img_numpy = img_numpy * 255
        img_numpy = img_numpy.astype(np.uint8)
        if img_numpy.shape[2] == 3:
            img_opencv = cv.cvtColor(img_numpy, cv.COLOR_RGB2BGR)
        else:
            img_opencv = img_numpy

        return img_opencv


class FeaturesConverter(Converter):
    def _to_cv(self, data):
        keypoints = data.get('keypoints')
        descriptors = data.get('descriptors')

        keypoints_np = keypoints.detach().cpu().numpy()

        if keypoints_np.ndim == 3:
            keypoints_np = keypoints_np.reshape(-1, 2)

        keypoints_np = keypoints_np.astype(np.float32)

        keypoints = cv.KeyPoint_convert(keypoints_np)
        descriptors = descriptors.detach().cpu().numpy()

        result = {'kp': keypoints, 'des': descriptors}

        if 'width' in data:
            result['width'] = data['width']
        if 'height' in data:
            result['height'] = data['height']

        return result

    def _to_tensor(self, data, device='cpu'):
        kp = data.get('kp')
        des = data.get('des')

        keypoints_np = cv.KeyPoint_convert(kp)
        keypoints = torch.from_numpy(keypoints_np).to(device)
        descriptors = torch.from_numpy(des).to(device)

        result = {'keypoints': keypoints, 'descriptors': descriptors}

        if 'width' in data:
            result['width'] = data['width']
        if 'height' in data:
            result['height'] = data['height']

        return result


class MatchesConverter(Converter):
    def _to_cv(self, data):
        matches = data.get('matches')
        matches_np = matches.detach().cpu().numpy()

        dmatches = []
        for query_idx, train_idx in matches_np:
            dmatch = cv.DMatch()
            dmatch.queryIdx = int(query_idx)
            dmatch.trainIdx = int(train_idx)
            dmatches.append(dmatch)

        return {'matches': dmatches}

    def _to_tensor(self, data, device='cpu'):
        dmatches = data.get('matches')
        matches_np = np.array([[dmatch.queryIdx, dmatch.trainIdx] for dmatch in dmatches])
        matches = torch.from_numpy(matches_np).to(device)
        return {'matches': matches}
