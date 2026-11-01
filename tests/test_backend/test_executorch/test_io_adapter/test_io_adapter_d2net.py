from unittest.mock import patch

import cv2 as cv
import numpy as np
import pytest
import torch

from src.backend.executorch.io_adapter.d2net import D2NetExecuTorchIOAdapter
from src.backend.io_adapter_base import IOAdapter


@pytest.fixture
def adapter():
    return D2NetExecuTorchIOAdapter("d2net_executorch", config={"input_shape": (1, 3, 4, 6)})


class TestD2NetExecuTorchIOAdapterRegistration:
    def test_registered_in_factory(self):
        assert IOAdapter._METHODS["d2net_executorch"] is D2NetExecuTorchIOAdapter


class TestD2NetExecuTorchIOAdapterPreprocess:
    def test_preprocess_uses_caffe_bgr_normalization(self, adapter):
        image = np.array([[[10, 20, 30]]], dtype=np.uint8)

        result = adapter.preprocess({"image": image})

        expected = torch.tensor([10 - 103.939, 20 - 116.779, 30 - 123.68])
        torch.testing.assert_close(result["image"][0, :, 0, 0], expected)
        assert result["image"].shape == (1, 3, 4, 6)

    def test_preprocess_preserves_extra_inputs(self, adapter):
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        result = adapter.preprocess({"image": image, "threshold": 0.2})
        assert result["threshold"] == 0.2

    @pytest.mark.parametrize("inputs", [{}, [], None])
    def test_preprocess_requires_image_dictionary(self, adapter, inputs):
        with pytest.raises(ValueError):
            adapter.preprocess(inputs)

    def test_preprocess_rejects_non_numpy_image(self, adapter):
        with pytest.raises(TypeError):
            adapter.preprocess({"image": torch.zeros(1, 3, 4, 6)})


class TestD2NetExecuTorchIOAdapterPostprocess:
    def test_postprocess_converts_and_scales_features(self, adapter):
        adapter._original_height = 8
        adapter._original_width = 12
        adapter._processed_height = 4
        adapter._processed_width = 6
        extracted = (
            torch.tensor([[1.0, 2.0], [3.0, 1.0]]),
            torch.tensor([[0.5, 0.5], [0.0, 1.0]]),
            torch.tensor([0.9, 0.8]),
        )

        with patch.object(adapter, "_extract_features", return_value=extracted):
            result = adapter.postprocess([torch.zeros(1, 2, 2, 2)])

        assert all(isinstance(keypoint, cv.KeyPoint) for keypoint in result["keypoints"])
        assert result["keypoints"][0].pt == (2.0, 4.0)
        np.testing.assert_array_equal(result["descriptors"], extracted[1].numpy())
        np.testing.assert_array_equal(result["scores"], extracted[2].numpy())

    @pytest.mark.parametrize("outputs", [torch.zeros(2, 3), torch.zeros(2, 3, 4, 5), "output"])
    def test_postprocess_rejects_invalid_dense_output(self, adapter, outputs):
        with pytest.raises(ValueError, match="dense output"):
            adapter.postprocess(outputs)

    def test_extract_features_returns_empty_result_without_detections(self, adapter):
        adapter._detection = lambda dense: torch.zeros_like(dense, dtype=torch.bool)
        dense = torch.zeros(1, 4, 3, 3)

        keypoints, descriptors, scores = adapter._extract_features(dense)

        assert keypoints.shape == (0, 2)
        assert descriptors.shape == (0, 4)
        assert scores.shape == (0,)

    def test_interpolate_descriptors_filters_out_of_bounds_keypoints(self):
        dense = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
        keypoints = torch.tensor([[1.5, -1.0], [1.5, 1.0]])

        descriptors, valid_ids = D2NetExecuTorchIOAdapter._interpolate_descriptors(
            keypoints, dense
        )

        assert valid_ids.tolist() == [0]
        torch.testing.assert_close(descriptors[:, 0], torch.tensor([6.0, 15.0]))
