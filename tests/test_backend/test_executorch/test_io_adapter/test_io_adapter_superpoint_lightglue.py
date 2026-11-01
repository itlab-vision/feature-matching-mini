from unittest.mock import patch

import cv2 as cv
import numpy as np
import pytest
import torch

from src.backend.executorch.io_adapter.superpoint_lightglue import (
    SuperPointLightGlueExecuTorchIOAdapter,
)
from src.backend.io_adapter_base import IOAdapter


@pytest.fixture
def adapter():
    return SuperPointLightGlueExecuTorchIOAdapter(
        "superpoint_lightglue_executorch",
        config={"input_shape": (1, 3, 8, 16), "num_keypoints": 2, "nms_radius": 3},
    )


class TestSuperPointLightGlueExecuTorchIOAdapterRegistration:
    def test_registered_in_factory(self):
        assert (
            IOAdapter._METHODS["superpoint_lightglue_executorch"]
            is SuperPointLightGlueExecuTorchIOAdapter
        )


class TestSuperPointLightGlueExecuTorchIOAdapterPreprocess:
    def test_preprocess_records_original_and_input_size(self, adapter):
        result = adapter.preprocess({"image": np.zeros((4, 8, 3), dtype=np.uint8)})
        assert result["image"].shape == (1, 3, 8, 16)
        assert (adapter._original_height, adapter._original_width) == (4, 8)
        assert (adapter._input_height, adapter._input_width) == (8, 16)

    @pytest.mark.parametrize("inputs", [{}, [], None])
    def test_preprocess_requires_image_dictionary(self, adapter, inputs):
        with pytest.raises(ValueError):
            adapter.preprocess(inputs)

    def test_preprocess_rejects_non_numpy_image(self, adapter):
        with pytest.raises(TypeError):
            adapter.preprocess({"image": torch.zeros(1, 3, 8, 16)})


class TestSuperPointLightGlueExecuTorchIOAdapterPostprocess:
    @patch("src.backend.executorch.io_adapter.superpoint_lightglue.sample_descriptors")
    @patch("src.backend.executorch.io_adapter.superpoint_lightglue.nms_topk")
    def test_postprocess_extracts_features(self, mock_nms, mock_sample, adapter):
        mock_nms.return_value = (torch.tensor([[4.0, 2.0], [8.0, 4.0]]), torch.tensor([0.9, 0.7]))
        mock_sample.return_value = torch.ones(2, 4)
        probabilities = torch.zeros(1, 65, 1, 2)
        dense = torch.zeros(1, 4, 1, 2)

        result = adapter.postprocess((torch.empty(0), probabilities, dense))

        assert all(isinstance(keypoint, cv.KeyPoint) for keypoint in result["keypoints"])
        assert result["descriptors"].shape == (2, 4)
        mock_nms.assert_called_once()
        assert mock_nms.call_args.args[1:] == (2, 3)
        torch.testing.assert_close(mock_sample.call_args.args[1], mock_nms.return_value[0] / 8)

    def test_postprocess_requires_three_outputs(self, adapter):
        with pytest.raises(ValueError, match="SuperPoint outputs"):
            adapter.postprocess([torch.zeros(1)])

    def test_postprocess_requires_65_probability_channels(self, adapter):
        with pytest.raises(ValueError, match="65"):
            adapter.postprocess((torch.empty(0), torch.zeros(1, 64, 2, 2), torch.zeros(1, 4, 2, 2)))

    def test_postprocess_validates_spatial_shape(self, adapter):
        with pytest.raises(ValueError, match="same spatial shape"):
            adapter.postprocess((torch.empty(0), torch.zeros(1, 65, 2, 2), torch.zeros(1, 4, 1, 1)))
