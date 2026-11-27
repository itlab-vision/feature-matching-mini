import cv2 as cv
import numpy as np

from src.backend.io_adapter_base import IOAdapter
from src.backend.lightglue_utils import keypoints_to_numpy, normalize_keypoints


@IOAdapter.register('lightglue')
@IOAdapter.register('lightglue_disk')
@IOAdapter.register('lightglue_superpoint')
class LightGlueIOAdapter(IOAdapter):
    def _features(self, features):
        points = features.get('keypoints', features.get('kp'))
        descriptors = features.get('descriptors', features.get('des'))
        if points is None or descriptors is None:
            raise ValueError('LightGlue requires keypoints and descriptors')
        result = dict(features)
        result['keypoints'] = keypoints_to_numpy(points)
        descriptors = np.asarray(descriptors, dtype=np.float32)
        result['descriptors'] = descriptors[0] if descriptors.ndim == 3 and descriptors.shape[0] == 1 else descriptors
        if features.get('scores') is not None:
            result['scores'] = np.asarray(features['scores'], dtype=np.float32)
        return result

    def preprocess(self, inputs):
        features0 = self._features(inputs['features0'])
        features1 = self._features(inputs['features1'])
        if any(len(features['keypoints']) == 0 or len(features['descriptors']) == 0
               for features in (features0, features1)):
            return None
        descriptor = self._config.get('descriptor_class')
        descriptor_name = descriptor._descriptor_name if descriptor is not None else ''
        result = {**inputs, 'features0': features0, 'features1': features1,
                  'add_scale_ori': descriptor_name in ('sift_lightglue', 'doghardnet_lightglue')}
        for features in (features0, features1):
            points = features['keypoints'].copy()
            name = features.get('extractor_name', '').lower()
            height, width = features.get('height'), features.get('width')
            if self._config.get('normalize_keypoints', False):
                points = normalize_keypoints(points, height, width)
            elif 'disk' in name and height and width and min(height, width) > 1:
                points[:, 0] /= width - 1
                points[:, 1] /= height - 1
            elif 'aliked' in name and height and width and min(height, width) > 1:
                points[:, 0] = 2 * points[:, 0] / width - 1
                points[:, 1] = 2 * points[:, 1] / height - 1
            features['normalized_keypoints'] = points.astype(np.float32)
        return result

    def postprocess(self, outputs):
        if outputs is None:
            return {'matches': np.empty(0, dtype=object), 'scores': np.empty(0, dtype=np.float32)}
        result = dict(outputs)
        matches = np.asarray(result['matches'], dtype=np.int64).reshape(-1, 2)
        scores = result['scores'] = np.asarray(result['scores'], dtype=np.float32).reshape(-1)
        result['matches'] = np.asarray([
            cv.DMatch(int(i), int(j), 0, max(0.0, 1.0 - float(score))) for (i, j), score in zip(matches, scores)
        ], dtype=object)
        return result
