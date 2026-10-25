from src.backend.io_adapter_base import IOAdapter
from src.backend.torch.io_adapter.io_adapter import TorchIOAdapter
from src.backend.torch.utils import features_to_tensor, matches_to_cv

from thirdparty.lightglue.lightglue.utils import rbd


@IOAdapter.register("lightglue_torch")
class LightGlueTorchIOAdapter(TorchIOAdapter):
    def preprocess(self, inputs):
        keypoints0 = inputs['features0'].get('keypoints')
        descriptors0 = inputs['features0'].get('descriptors')
        scores0 = inputs['features0'].get('scores')
        data0 = {'keypoints': keypoints0, 'descriptors': descriptors0, 'scores': scores0}

        keypoints1 = inputs['features1'].get('keypoints')
        descriptors1 = inputs['features1'].get('descriptors')
        scores1 = inputs['features1'].get('scores')
        data1 = {'keypoints': keypoints1, 'descriptors': descriptors1, 'scores': scores1}

        features0 = features_to_tensor(data0)
        features1 = features_to_tensor(data1)

        return {
            "image0": {
                "keypoints": features0.pop('keypoints').unsqueeze(0).to(self._device),
                "descriptors": features0.pop('descriptors').unsqueeze(0).to(self._device),
                "keypoint_scores": features0.pop('scores').unsqueeze(0).to(self._device),
                "scales": inputs['features0'].get('scales'),
                "oris": inputs['features0'].get('oris'),
                "image_size": inputs['features0'].get('image_size'),
            },
            "image1": {
                "keypoints": features1.pop('keypoints').unsqueeze(0).to(self._device),
                "descriptors": features1.pop('descriptors').unsqueeze(0).to(self._device),
                "keypoint_scores": features1.pop('scores').unsqueeze(0).to(self._device),
                "scales": inputs['features1'].get('scales'),
                "oris": inputs['features1'].get('oris'),
                "image_size": inputs['features1'].get('image_size'),
            }
        }

    def postprocess(self, outputs):
        matches = rbd(outputs)
        return matches_to_cv(matches)
