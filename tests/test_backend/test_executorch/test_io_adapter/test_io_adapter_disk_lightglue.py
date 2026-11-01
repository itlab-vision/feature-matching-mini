from unittest.mock import patch

import cv2 as cv
import numpy as np
import pytest
import torch

from src.backend.executorch.io_adapter.disk_lightglue import DiskLightGlueExecuTorchIOAdapter
from src.backend.io_adapter_base import IOAdapter


@pytest.fixture
def adapter():
    return DiskLightGlueExecuTorchIOAdapter(
        "disk_lightglue_executorch",
        config={"input_shape": (1, 3, 4, 8), "num_keypoints": 2, "nms_radius": 3},
    )


class TestDiskLightGlueExecuTorchIOAdapterRegistration:
    def test_registered_in_factory(self):
        assert IOAdapter._METHODS["disk_lightglue_executorch"] is DiskLightGlueExecuTorchIOAdapter


class TestDiskLightGlueExecuTorchIOAdapterPreprocess:
    def test_preprocess_keeps_pixel_scale_expected_by_disk(self, adapter):
        image = np.full((4, 8, 3), 255, dtype=np.uint8)
        result = adapter.preprocess({"image": image})
        torch.testing.assert_close(result["image"], torch.full((1, 3, 4, 8), 255.0))

    @pytest.mark.parametrize("inputs", [{}, [], None])
    def test_preprocess_requires_image_dictionary(self, adapter, inputs):
        with pytest.raises(ValueError):
            adapter.preprocess(inputs)

    def test_preprocess_rejects_non_numpy_image(self, adapter):
        with pytest.raises(TypeError):
            adapter.preprocess({"image": torch.zeros(1, 3, 4, 8)})


class TestDiskLightGlueExecuTorchIOAdapterPostprocess:
    @patch("src.backend.executorch.io_adapter.disk_lightglue.sample_descriptors")
    @patch("src.backend.executorch.io_adapter.disk_lightglue.nms_topk")
    def test_postprocess_extracts_and_scales_features(self, mock_nms, mock_sample, adapter):
        adapter._original_height = 8
        adapter._original_width = 16
        adapter._input_height = 4
        adapter._input_width = 8
        mock_nms.return_value = (torch.tensor([[1.0, 2.0], [3.0, 1.0]]), torch.tensor([0.9, 0.7]))
        mock_sample.return_value = torch.ones(2, 4)

        result = adapter.postprocess((torch.zeros(1, 1, 4, 8), torch.zeros(1, 4, 4, 8)))

        assert all(isinstance(keypoint, cv.KeyPoint) for keypoint in result["keypoints"])
        assert result["keypoints"][0].pt == (2.0, 4.0)
        assert result["descriptors"].shape == (2, 4)
        mock_nms.assert_called_once()
        assert mock_nms.call_args.args[1:] == (2, 3)

    @pytest.mark.parametrize(
        "outputs",
        [torch.zeros(1, 1, 2, 2), [torch.zeros(1, 1, 2, 2)], [torch.zeros(1), torch.zeros(1), torch.zeros(1)]],
    )
    def test_postprocess_requires_two_outputs(self, adapter, outputs):
        with pytest.raises(ValueError, match="must return"):
            adapter.postprocess(outputs)

    def test_postprocess_validates_heatmap_shape(self, adapter):
        with pytest.raises(ValueError, match="heatmap"):
            adapter.postprocess((torch.zeros(1, 2, 4, 4), torch.zeros(1, 4, 4, 4)))

    def test_postprocess_validates_spatial_shape(self, adapter):
        with pytest.raises(ValueError, match="same spatial shape"):
            adapter.postprocess((torch.zeros(1, 1, 4, 4), torch.zeros(1, 4, 2, 2)))
