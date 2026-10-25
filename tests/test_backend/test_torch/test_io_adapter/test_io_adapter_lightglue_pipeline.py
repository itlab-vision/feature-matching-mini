import pytest
import torch

from unittest.mock import MagicMock, patch

from src.backend.torch.io_adapter.lightglue_pipeline import (
    LightGlueFeatureExtractorTorchIOAdapter,
)


@pytest.fixture
def adapter():
    return LightGlueFeatureExtractorTorchIOAdapter("superpoint")


class TestLightGlueFeatureExtractorTorchIOAdapter:
    def test_registered(self):
        assert hasattr(LightGlueFeatureExtractorTorchIOAdapter, "preprocess")
        assert hasattr(LightGlueFeatureExtractorTorchIOAdapter, "postprocess")

    def test_extractor_classes(self, adapter):
        assert "superpoint_lightglue_torch" in adapter._EXTRACTOR_CLASSES
        assert "disk_lightglue_torch" in adapter._EXTRACTOR_CLASSES
        assert "sift_lightglue_torch" in adapter._EXTRACTOR_CLASSES
        assert "aliked_lightglue_torch" in adapter._EXTRACTOR_CLASSES
        assert "doghardnet_lightglue_torch" in adapter._EXTRACTOR_CLASSES

    @patch("src.backend.torch.io_adapter.lightglue_pipeline.image_to_tensor")
    def test_preprocess(self, mock_image_to_tensor, adapter):
        image = MagicMock()
        image_tensor = torch.rand(1, 3, 32, 32)
        mock_image_to_tensor.return_value = image_tensor

        inputs = {
            "image": image,
            "custom_param": 123,
        }

        result = adapter.preprocess(inputs)

        mock_image_to_tensor.assert_called_once_with(image)
        assert result["image"] is image_tensor
        assert result["custom_param"] == 123
        assert "image" not in inputs

    @patch("src.backend.torch.io_adapter.lightglue_pipeline.features_to_cv")
    def test_postprocess(self, mock_features_to_cv, adapter):
        keypoints = torch.rand(1, 10, 2)
        descriptors = torch.rand(1, 10, 128)
        scores = torch.rand(1, 10)
        extra_output = torch.tensor([1, 2, 3])

        mock_features_to_cv.return_value = {
            "keypoints": "cv_keypoints",
            "descriptors": "cv_descriptors",
            "scores": "cv_scores",
        }

        outputs = {
            "keypoints": keypoints,
            "descriptors": descriptors,
            "keypoint_scores": scores,
            "extra": extra_output,
        }

        result = adapter.postprocess(outputs)

        mock_features_to_cv.assert_called_once()
        features = mock_features_to_cv.call_args.args[0]

        assert torch.equal(features["keypoints"], keypoints.squeeze(0))
        assert torch.equal(features["descriptors"], descriptors.squeeze(0))
        assert torch.equal(features["scores"], scores.squeeze(0))

        assert result["keypoints"] == "cv_keypoints"
        assert result["descriptors"] == "cv_descriptors"
        assert result["scores"] == "cv_scores"
        assert result["extra"] is extra_output

    @patch("src.backend.torch.io_adapter.lightglue_pipeline.features_to_cv")
    def test_postprocess_without_extra_outputs(self, mock_features_to_cv, adapter):
        mock_features_to_cv.return_value = {"converted": True}

        result = adapter.postprocess({
            "keypoints": torch.rand(1, 5, 2),
            "descriptors": torch.rand(1, 5, 64),
            "keypoint_scores": torch.rand(1, 5),
        })

        assert result == {"converted": True}

    @patch("src.backend.torch.io_adapter.lightglue_pipeline.features_to_cv")
    def test_postprocess_empty_features(self, mock_features_to_cv, adapter):
        mock_features_to_cv.return_value = {}

        result = adapter.postprocess({
            "keypoints": torch.empty(1, 0, 2),
            "descriptors": torch.empty(1, 0, 128),
            "keypoint_scores": torch.empty(1, 0),
        })

        assert result == {}
        mock_features_to_cv.assert_called_once()
