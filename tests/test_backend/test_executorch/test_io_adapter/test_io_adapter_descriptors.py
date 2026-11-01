import cv2 as cv
import numpy as np
import pytest
import torch

from src.backend.executorch.io_adapter.descriptors import PatchDescriptorExecuTorchIOAdapter
from src.backend.io_adapter_base import IOAdapter


@pytest.fixture
def adapter():
    return PatchDescriptorExecuTorchIOAdapter("hardnet_executorch", config={"patch_size": 8})


class TestPatchDescriptorExecuTorchIOAdapterRegistration:
    @pytest.mark.parametrize("name", ["hardnet_executorch", "tfeat_executorch"])
    def test_registered_aliases(self, name):
        assert IOAdapter._METHODS[name] is PatchDescriptorExecuTorchIOAdapter


class TestPatchDescriptorExecuTorchIOAdapterPreprocess:
    def test_preprocess_extracts_patch_for_cv_keypoint(self, adapter):
        image = np.full((16, 16), 255, dtype=np.uint8)
        keypoints = np.asarray([cv.KeyPoint(8, 8, 1)], dtype=object)

        result = adapter.preprocess({"image": image, "keypoints": keypoints})

        assert result["patches"].shape == (1, 1, 8, 8)
        assert result["patches"].dtype == torch.float32
        torch.testing.assert_close(result["patches"], torch.ones_like(result["patches"]))

    def test_preprocess_accepts_numeric_keypoints_and_scores(self, adapter):
        image = torch.zeros(1, 3, 16, 16)
        scores = np.array([0.7, 0.5], dtype=np.float32)

        result = adapter.preprocess(
            {"image": image, "kp": np.array([[4, 4], [10, 10]]), "scores": scores}
        )

        assert result["patches"].shape == (2, 1, 8, 8)
        np.testing.assert_array_equal(adapter._scores, scores)

    def test_preprocess_empty_keypoints_returns_empty_batch(self, adapter):
        result = adapter.preprocess(
            {"image": np.zeros((16, 16), dtype=np.uint8), "keypoints": []}
        )
        assert result["patches"].shape == (0, 1, 8, 8)

    @pytest.mark.parametrize("inputs", [{}, {"image": np.zeros((4, 4), dtype=np.uint8)}])
    def test_preprocess_requires_image_and_keypoints(self, adapter, inputs):
        with pytest.raises(ValueError):
            adapter.preprocess(inputs)

    def test_preprocess_rejects_mismatched_scores(self, adapter):
        with pytest.raises(ValueError, match="scores"):
            adapter.preprocess(
                {
                    "image": np.zeros((16, 16), dtype=np.uint8),
                    "keypoints": [[4, 4], [8, 8]],
                    "scores": [1.0],
                }
            )

    def test_prepare_image_rejects_unsupported_value(self):
        with pytest.raises(TypeError):
            PatchDescriptorExecuTorchIOAdapter._prepare_image("image")


class TestPatchDescriptorExecuTorchIOAdapterPostprocess:
    def test_postprocess_returns_descriptors_and_saved_features(self, adapter):
        keypoints = np.asarray([cv.KeyPoint(3, 4, 1)], dtype=object)
        adapter.preprocess({"image": np.zeros((8, 8), dtype=np.uint8), "keypoints": keypoints})
        descriptors = torch.tensor([[1.0, 2.0]])

        result = adapter.postprocess((descriptors,))

        assert result["keypoints"] is adapter._keypoints
        np.testing.assert_array_equal(result["descriptors"], descriptors.numpy())
        np.testing.assert_array_equal(result["scores"], np.ones(1, dtype=np.float32))

    @pytest.mark.parametrize(
        "outputs",
        [(torch.zeros(1, 2), torch.zeros(1, 2)), torch.zeros(1, 2, 3), "output"],
    )
    def test_postprocess_rejects_invalid_outputs(self, adapter, outputs):
        with pytest.raises(ValueError):
            adapter.postprocess(outputs)

    def test_postprocess_rejects_descriptor_count_mismatch(self, adapter):
        adapter._keypoints = np.empty(2, dtype=object)
        with pytest.raises(ValueError, match="number of descriptors"):
            adapter.postprocess(torch.zeros(1, 128))
