from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from src.backend.torch.model_wrapper.lightglue_pipeline import LightGlueFeatureExtractorTorchModelWrapper


@pytest.fixture(autouse=True)
def clear_shared_models():
    LightGlueFeatureExtractorTorchModelWrapper._shared_models.clear()
    yield
    LightGlueFeatureExtractorTorchModelWrapper._shared_models.clear()


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def wrapper(mock_logger):
    model_wrapper = (LightGlueFeatureExtractorTorchModelWrapper.__new__(LightGlueFeatureExtractorTorchModelWrapper))

    model_wrapper._logger = mock_logger
    model_wrapper._device = torch.device("cpu")
    model_wrapper._config = {}
    model_wrapper._model_name = "superpoint_lightglue"
    model_wrapper._nfeatures = 10
    model_wrapper._model = None

    model_wrapper._EXTRACTOR_CLASSES = dict(LightGlueFeatureExtractorTorchModelWrapper._EXTRACTOR_CLASSES)
    return model_wrapper


class TestLightGlueFeatureExtractorTorchModelWrapper:
    def test_load(self, wrapper):
        mock_superpoint = MagicMock()
        mock_model = MagicMock()

        mock_superpoint.return_value.eval.return_value.to.return_value = (mock_model)
        wrapper._EXTRACTOR_CLASSES["superpoint_lightglue_torch"] = mock_superpoint
        result = wrapper.load()

        mock_superpoint.assert_called_once_with()
        mock_superpoint.return_value.eval.assert_called_once_with()
        mock_superpoint.return_value.eval.return_value.to.assert_called_once_with(torch.device("cpu"))

        assert result["model"] is mock_model
        assert wrapper._model is mock_model

    def test_load_disk_extractor(self, wrapper):
        mock_disk = MagicMock()
        mock_model = MagicMock()

        wrapper._model_name = "disk_lightglue"
        wrapper._config = {"max_num_keypoints": 512}
        wrapper._EXTRACTOR_CLASSES["disk_lightglue_torch"] = mock_disk
        mock_disk.return_value.eval.return_value.to.return_value = mock_model

        result = wrapper.load()

        mock_disk.assert_called_once_with(max_num_keypoints=512)
        mock_disk.return_value.eval.assert_called_once_with()
        mock_disk.return_value.eval.return_value.to.assert_called_once_with(torch.device("cpu"))
        assert result["model"] is mock_model

    @pytest.mark.parametrize("extractor_name", [
        "superpoint_lightglue",
        "disk_lightglue",
        "aliked_lightglue",
        "sift_lightglue",
        "doghardnet_lightglue",
    ])
    def test_load_forwards_max_num_keypoints(self, wrapper, extractor_name):
        mock_extractor = MagicMock()
        mock_model = MagicMock()

        wrapper._model_name = extractor_name
        wrapper._config = {"max_num_keypoints": 256}
        wrapper._EXTRACTOR_CLASSES[f"{extractor_name}_torch"] = mock_extractor
        mock_extractor.return_value.eval.return_value.to.return_value = mock_model

        result = wrapper.load()

        mock_extractor.assert_called_once_with(max_num_keypoints=256)
        assert result["model"] is mock_model

    def test_load_nfeatures_not_mapped_for_other_extractors(self, wrapper):
        mock_roma = MagicMock()
        mock_model = MagicMock()

        wrapper._model_name = "roma"
        wrapper._config = {"nfeatures": 128}
        wrapper._EXTRACTOR_CLASSES["roma_torch"] = mock_roma
        mock_roma.return_value.eval.return_value.to.return_value = mock_model

        result = wrapper.load()

        mock_roma.assert_called_once_with(nfeatures=128)
        assert result["model"] is mock_model

    def test_load_unknown_extractor(self, wrapper):
        wrapper._model_name = "unknown_extractor"

        with pytest.raises(ValueError, match="Extractor 'unknown_extractor' not found"):
            wrapper.load()

    def test_load_reuses_shared_model(self, wrapper):
        mock_superpoint = MagicMock()
        mock_model = MagicMock()

        mock_superpoint.return_value.eval.return_value.to.return_value = (mock_model)
        wrapper._EXTRACTOR_CLASSES["superpoint_lightglue_torch"] = mock_superpoint

        result1 = wrapper.load()
        result2 = wrapper.load()

        mock_superpoint.assert_called_once_with()
        assert result1["model"] is mock_model
        assert result2["model"] is mock_model

    def test_load_separate_models_for_different_devices(self, wrapper):
        mock_superpoint = MagicMock()

        first_extractor = MagicMock()
        second_extractor = MagicMock()
        first_model = MagicMock()
        second_model = MagicMock()

        first_extractor.eval.return_value.to.return_value = first_model
        second_extractor.eval.return_value.to.return_value = second_model

        mock_superpoint.side_effect = [
            first_extractor,
            second_extractor,
        ]
        wrapper._EXTRACTOR_CLASSES["superpoint_lightglue_torch"] = mock_superpoint

        cpu_device = MagicMock()
        cpu_device.type = "cpu"

        cuda_device = MagicMock()
        cuda_device.type = "cuda"

        wrapper._device = cpu_device
        result1 = wrapper.load()

        wrapper._device = cuda_device
        result2 = wrapper.load()

        assert mock_superpoint.call_count == 2
        first_extractor.eval.return_value.to.assert_called_once_with(cpu_device)
        second_extractor.eval.return_value.to.assert_called_once_with(cuda_device)
        assert result1["model"] is first_model
        assert result2["model"] is second_model

    def test_call_converts_image_and_extracts_numpy_features(self, wrapper):
        wrapper._model = MagicMock()
        wrapper._model.extract.return_value = {
            "keypoints": torch.zeros(1, 3, 2),
            "descriptors": torch.ones(1, 3, 4),
            "keypoint_scores": torch.ones(1, 3),
        }
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        image[:, :, 2] = 255

        result = wrapper.call({"image": image})

        tensor = wrapper._model.extract.call_args.args[0]
        assert tensor.shape == (1, 3, 8, 10)
        torch.testing.assert_close(tensor[0, :, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
        assert result["keypoints"].shape == (3, 2)
        np.testing.assert_array_equal(result["descriptors"], np.ones((3, 4)))
        np.testing.assert_array_equal(result["scores"], np.ones(3))

    def test_call_logs_and_reraises_exception(self, wrapper):
        wrapper._model = MagicMock()
        wrapper._model.extract.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            wrapper.call({"image": np.zeros((8, 10, 3), dtype=np.uint8)})
        wrapper._logger.error.assert_called_once_with("LightGlue extractor inference error: boom")
