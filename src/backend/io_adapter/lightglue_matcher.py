import cv2 as cv
import numpy as np

from src.backend.io_adapter_base import IOAdapter


@IOAdapter.register('lightglue')
@IOAdapter.register('lightglue_disk')
@IOAdapter.register('lightglue_superpoint')
class LightGlueIOAdapter(IOAdapter):
    def _features(self, features):
        points = features.get('keypoints', features.get('kp'))
        descriptors = features.get('descriptors', features.get('des'))
        if points is None or descriptors is None:
            raise ValueError('LightGlue requires keypoints and descriptors')
        if len(points) and isinstance(points[0], cv.KeyPoint):
            points = cv.KeyPoint_convert(points)
        result = dict(features)
        result['keypoints'] = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        result['descriptors'] = np.asarray(descriptors, dtype=np.float32)
        if features.get('scores') is not None:
            result['scores'] = np.asarray(features['scores'], dtype=np.float32)
        return result

    def preprocess(self, inputs):
        features0 = self._features(inputs['features0'])
        features1 = self._features(inputs['features1'])
        result = {**inputs, 'features0': features0, 'features1': features1}
        for index, features in enumerate((features0, features1)):
            points = features['keypoints'].copy()
            name = features.get('extractor_name', '').lower()
            height, width = features.get('height'), features.get('width')
            if 'disk' in name:
                points[:, 0] /= width - 1
                points[:, 1] /= height - 1
            elif 'aliked' in name:
                points[:, 0] = 2 * points[:, 0] / width - 1
                points[:, 1] = 2 * points[:, 1] / height - 1
            features['normalized_keypoints'] = points.astype(np.float32)
        return result

    def postprocess(self, outputs):
        if outputs is None:
            return {'matches': np.empty(0, dtype=object), 'scores': np.empty(0, dtype=np.float32)}
        result = dict(outputs)
        matches = np.asarray(result['matches'], dtype=np.int64).reshape(-1, 2)
        result['matches'] = np.asarray([cv.DMatch(int(i), int(j), 0) for i, j in matches])
        result['scores'] = np.asarray(result['scores'], dtype=np.float32).reshape(-1)
        return result
