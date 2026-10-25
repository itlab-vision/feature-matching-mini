from src.backend.io_adapter_base import IOAdapter
from src.backend.torch.io_adapter.io_adapter import TorchIOAdapter
from src.backend.torch.utils import image_to_tensor, features_to_cv


@IOAdapter.register("light_glue_feature_extractor_torch")
class LightGlueFeatureExtractorTorchIOAdapter(TorchIOAdapter):
    _EXTRACTOR_CLASSES = {
        'superpoint_lightglue_torch',
        'disk_lightglue_torch',
        'sift_lightglue_torch',
        'aliked_lightglue_torch',
        'doghardnet_lightglue_torch'
    }

    def preprocess(self, inputs):
        img = inputs.pop('image')
        img = image_to_tensor(img)
        return {'image': img, **inputs}

    def postprocess(self, outputs):
        keypoints = outputs.pop('keypoints').squeeze(0)
        descriptors = outputs.pop('descriptors').squeeze(0)
        scores = outputs.pop('keypoint_scores').squeeze(0)

        data = {'keypoints': keypoints, 'descriptors': descriptors, 'scores': scores}
        data = features_to_cv(data)
        return {**data, **outputs}
