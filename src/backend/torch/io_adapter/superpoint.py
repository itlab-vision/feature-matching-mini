import torch

from src.backend.io_adapter_base import IOAdapter
from src.backend.torch.io_adapter.io_adapter import TorchIOAdapter
from src.backend.torch.utils import to_numpy_bgr, features_to_cv, image_to_tensor


@IOAdapter.register("superpoint_torch")
class SuperPointTorchIOAdapter(TorchIOAdapter):
    def preprocess(self, inputs):
        img = inputs.pop('image')
        processor = inputs.get('processor')

        img = image_to_tensor(img)
        input_type = 'torch' if isinstance(img, torch.Tensor) else 'numpy'
        img = to_numpy_bgr(img, input_type=input_type)
        height, width = img.shape[:2]
        img = processor(img, return_tensors="pt").to(self._device)
        return {'image': img, 'height': height, 'width': width, **inputs}

    def postprocess(self, outputs):
        processor = outputs.get('processor')
        inference_outputs = outputs.get('inference_outputs')
        height, width = outputs.get('height'), outputs.get('width')
        processed = processor.post_process_keypoint_detection(inference_outputs, [[height, width]])[0]

        outputs['keypoints'] = processed['keypoints']
        outputs['descriptors'] = processed['descriptors']
        outputs['scores'] = processed['scores']
        return features_to_cv(outputs)
