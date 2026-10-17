from unittest.mock import MagicMock

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

    def test_call_adds_batch_dimension(self, wrapper):
        image = torch.rand(1, 32, 32)
        mock_model = MagicMock()
        expected_output = {"keypoints": torch.rand(1, 10, 2)}
        mock_model.extract.return_value = expected_output

        wrapper._model = mock_model

        result = wrapper.call(
            {
                "image": image,
                "model": mock_model,
            }
        )

        mock_model.extract.assert_called_once()
        actual_image = mock_model.extract.call_args.args[0]

        assert actual_image.shape == (1, 1, 32, 32)
        assert actual_image.device.type == "cpu"
        assert result is expected_output

    def test_call_does_not_add_batch_dimension_to_batched_image(self, wrapper):
        image = torch.rand(1, 32, 32)
        mock_model = MagicMock()
        expected_output = {"keypoints": torch.rand(1, 10, 2)}
        mock_model.extract.return_value = expected_output

        wrapper._model = mock_model

        result = wrapper.call(
            {
                "image": image,
                "model": mock_model,
            }
        )

        actual_image = mock_model.extract.call_args.args[0]

        assert actual_image.shape == (1, 1, 32, 32)
        assert result is expected_output

    def test_call_logs_and_reraises_exception(self, wrapper):
        image = torch.rand(1, 32, 32)
        mock_model = MagicMock()
        mock_model.extract.side_effect = RuntimeError("inference failed")

        with pytest.raises(RuntimeError, match="inference failed"):
            wrapper.call(
                {
                    "image": image,
                    "model": mock_model,
                }
            )

        wrapper._logger.error.assert_called_once()
