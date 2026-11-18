import cv2 as cv
import numpy as np

from src.backend.io_adapter_base import IOAdapter


class PatchDescriptorIOAdapter(IOAdapter):
    def preprocess(self, inputs):
        if not isinstance(inputs, dict) or 'image' not in inputs:
            raise ValueError('Patch descriptor expects an image')
        keypoints = inputs.get('keypoints', inputs.get('kp'))
        if keypoints is None:
            raise ValueError('Patch descriptor expects keypoints')
        image = np.asarray(inputs['image'])
        if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (1, 3)):
            raise ValueError(f'Unsupported image shape: {image.shape}')
        keypoints = np.asarray(keypoints, dtype=object)
        if len(keypoints) and not isinstance(keypoints[0], cv.KeyPoint):
            raise TypeError('Patch descriptor expects OpenCV keypoints')
        scores = inputs.get('scores')
        scores = np.ones(len(keypoints), dtype=np.float32) if scores is None else np.asarray(scores, dtype=np.float32)
        if scores.shape != (len(keypoints),):
            raise ValueError('Keypoint scores must have shape [N]')
        return {**inputs, 'image': np.ascontiguousarray(image), 'keypoints': keypoints, 'scores': scores}

    def postprocess(self, outputs):
        result = dict(outputs)
        result['keypoints'] = np.asarray(result['keypoints'], dtype=object)
        descriptors = np.asarray(result['descriptors'], dtype=np.float32)
        result['descriptors'] = descriptors.reshape(len(result['keypoints']), descriptors.shape[-1])
        result['scores'] = np.asarray(result['scores'], dtype=np.float32)
        return result
