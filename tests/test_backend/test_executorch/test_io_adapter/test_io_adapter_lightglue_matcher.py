import cv2 as cv
import numpy as np
import pytest
import torch

from src.backend.executorch.io_adapter.lightglue_matcher import LightGlueExecuTorchIOAdapter
from src.backend.io_adapter_base import IOAdapter


@pytest.fixture
def adapter():
    return LightGlueExecuTorchIOAdapter("lightglue_executorch", config={"num_keypoints": 2})


@pytest.fixture
def features():
    return {
        "keypoints": np.asarray([cv.KeyPoint(1, 2, 1), cv.KeyPoint(3, 4, 1)], dtype=object),
        "descriptors": np.arange(8, dtype=np.float32).reshape(2, 4),
    }


class TestLightGlueExecuTorchIOAdapterRegistration:
    @pytest.mark.parametrize(
        "name",
        ["lightglue_executorch", "lightglue_disk_executorch", "lightglue_superpoint_executorch"],
    )
    def test_registered_aliases(self, name):
        assert IOAdapter._METHODS[name] is LightGlueExecuTorchIOAdapter


class TestLightGlueExecuTorchIOAdapterPreprocess:
    def test_preprocess_converts_cv_features_to_batched_tensors(self, adapter, features):
        result = adapter.preprocess({"features0": features, "features1": features})

        assert set(result) == {"keypoints0", "keypoints1", "descriptors0", "descriptors1"}
        assert result["keypoints0"].shape == (1, 2, 2)
        assert result["descriptors0"].shape == (1, 2, 4)
        torch.testing.assert_close(result["keypoints0"][0], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    def test_preprocess_accepts_tensor_and_legacy_feature_names(self, adapter):
        features = {"kp": torch.zeros(1, 2, 2), "des": torch.zeros(1, 2, 4)}
        result = adapter.preprocess({"features0": features, "features1": features})
        assert result["keypoints0"].shape == (1, 2, 2)
        assert result["descriptors1"].shape == (1, 2, 4)

    def test_preprocess_rejects_non_dictionary(self, adapter):
        with pytest.raises(TypeError):
            adapter.preprocess([])

    def test_preprocess_requires_both_feature_sets(self, adapter, features):
        with pytest.raises(ValueError, match="features1"):
            adapter.preprocess({"features0": features})

    def test_preprocess_validates_keypoint_count(self, adapter, features):
        invalid = {**features, "keypoints": np.array([[1.0, 2.0]], dtype=np.float32)}
        with pytest.raises(ValueError, match="expects 2 keypoints"):
            adapter.preprocess({"features0": invalid, "features1": features})

    def test_preprocess_validates_descriptor_shape(self, adapter, features):
        invalid = {**features, "descriptors": np.zeros((2, 4, 1), dtype=np.float32)}
        with pytest.raises(ValueError, match="descriptors must have shape"):
            adapter.preprocess({"features0": invalid, "features1": features})


class TestLightGlueExecuTorchIOAdapterPostprocess:
    def test_postprocess_keeps_only_valid_matches(self, adapter):
        outputs = (
            torch.tensor([[1, -1]]),
            torch.tensor([[0, -1]]),
            torch.tensor([[0.8, 0.1]]),
            torch.tensor([[0.8, 0.1]]),
        )

        result = adapter.postprocess(outputs)

        assert len(result["matches"]) == 1
        assert isinstance(result["matches"][0], cv.DMatch)
        assert result["matches"][0].queryIdx == 0
        assert result["matches"][0].trainIdx == 1
        np.testing.assert_array_equal(result["scores"], np.array([0.8], dtype=np.float32))

    def test_postprocess_requires_four_outputs(self, adapter):
        with pytest.raises(ValueError, match="LightGlue outputs"):
            adapter.postprocess([torch.zeros(1, 2)])

    @pytest.mark.parametrize("index", [0, 2])
    def test_postprocess_validates_match_and_score_shapes(self, adapter, index):
        outputs = [torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2)]
        outputs[index] = torch.zeros(2)
        with pytest.raises(ValueError):
            adapter.postprocess(outputs)
