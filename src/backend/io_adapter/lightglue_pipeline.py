import cv2 as cv
import numpy as np

from src.backend.io_adapter.aliked import ALIKEDIOAdapter
from src.backend.io_adapter.disk import DISKIOAdapter
from src.backend.io_adapter_base import IOAdapter
from src.backend.superpoint_dense import postprocess_superpoint_dense


class LightGlueFeatureIOAdapter(IOAdapter):
    def preprocess(self, inputs):
        if not isinstance(inputs, dict) or 'image' not in inputs:
            raise ValueError('LightGlue extractor expects an image')
        image = np.asarray(inputs['image'])
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f'Unsupported image shape: {image.shape}')
        self._image_size = image.shape[:2]
        return {**inputs, 'image': np.ascontiguousarray(image)}

    def postprocess(self, outputs):
        if not isinstance(outputs, dict):
            return self.postprocess_tensors(outputs)
        result = dict(outputs)
        points = np.asarray(result['keypoints'], dtype=np.float32).reshape(-1, 2)
        result['keypoints'] = np.asarray(cv.KeyPoint_convert(points), dtype=object)
        result['descriptors'] = np.asarray(result['descriptors'])
        result['scores'] = np.asarray(result['scores'])
        return result

    def postprocess_tensors(self, outputs):
        raise ValueError('Expected LightGlue features as a dictionary')


@IOAdapter.register('superpoint_lightglue')
class SuperPointLightGlueIOAdapter(LightGlueFeatureIOAdapter):
    def postprocess_tensors(self, outputs):
        shape = self._config.get('input_shape', (1, 3, 480, 640))
        count = self._config.get('num_keypoints', self._config.get('nfeatures', 512))
        return postprocess_superpoint_dense(
            outputs, self._image_size, shape[-2:], count, self._config.get('nms_radius', 4)
        )


@IOAdapter.register('disk_lightglue')
class DISKLightGlueIOAdapter(LightGlueFeatureIOAdapter):
    postprocess_tensors = DISKIOAdapter.postprocess


@IOAdapter.register('sift_lightglue')
class SIFTLightGlueIOAdapter(LightGlueFeatureIOAdapter):
    pass


@IOAdapter.register('aliked_lightglue')
class ALIKEDLightGlueIOAdapter(LightGlueFeatureIOAdapter):
    postprocess_tensors = ALIKEDIOAdapter.postprocess


@IOAdapter.register('doghardnet_lightglue')
class DoGHardNetLightGlueIOAdapter(LightGlueFeatureIOAdapter):
    pass
