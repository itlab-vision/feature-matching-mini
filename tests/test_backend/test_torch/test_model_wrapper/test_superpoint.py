import pytest
import torch

from unittest.mock import MagicMock, patch
from logging import Logger

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.torch.model_wrapper.superpoint import SuperPointTorchModelWrapper


@pytest.fixture
def mock_logger():
    return MagicMock(spec=Logger)


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.to.return_value = model
    return model


@pytest.fixture
def mock_processor():
    return MagicMock()


@pytest.fixture
def wrapper(mock_logger, mock_model, mock_processor):
    obj = SuperPointTorchModelWrapper.__new__(SuperPointTorchModelWrapper)
    obj._logger = mock_logger
    obj._device = torch.device("cpu")
    obj._config = {}
    obj._model = mock_model
    obj._image_processor = mock_processor
    return obj


@pytest.fixture
def inference_outputs():
    return {
        "keypoints": torch.rand(1, 100, 2),
        "descriptors": torch.rand(1, 100, 256),
        "scores": torch.rand(1, 100),
    }


class TestSuperPointTorchModelWrapperRegistration:

    def test_registered_in_factory(self):
        assert "superpoint_torch" in ModelWrapper._METHODS
        assert (ModelWrapper._METHODS["superpoint_torch"] is SuperPointTorchModelWrapper)


class TestSuperPointTorchModelWrapperLoad:
    @patch("src.backend.torch.model_wrapper.superpoint.AutoImageProcessor")
    @patch("src.backend.torch.model_wrapper.superpoint.SuperPointForKeypointDetection")
    def test_load_uses_local_checkpoint(self, mock_model_class, mock_processor_class, wrapper, mock_model,
                                        mock_processor, tmp_path):
        checkpoint = tmp_path / "superpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("{}")
        wrapper._config = {"checkpoint": str(checkpoint)}

        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        result = wrapper.load()

        mock_processor_class.from_pretrained.assert_called_once_with(str(checkpoint), local_files_only=True)
        mock_model_class.from_pretrained.assert_called_once_with(str(checkpoint), local_files_only=True)
        assert result == {"model": mock_model, "processor": mock_processor}

    @patch("src.backend.torch.model_wrapper.superpoint.AutoImageProcessor")
    @patch("src.backend.torch.model_wrapper.superpoint.SuperPointForKeypointDetection")
    def test_load_uses_remote_checkpoint_when_local_missing(self, mock_model_class, mock_processor_class, wrapper,
                                                            mock_model, mock_processor, tmp_path):
        checkpoint = tmp_path / "missing"
        wrapper._config = {"checkpoint": str(checkpoint)}
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model

        result = wrapper.load()
        remote_repo = "magic-leap-community/superpoint"
        mock_processor_class.from_pretrained.assert_called_once_with(remote_repo, local_files_only=False)
        mock_model_class.from_pretrained.assert_called_once_with(remote_repo, local_files_only=False)
        assert result["model"] is mock_model
        assert result["processor"] is mock_processor

    @patch("src.backend.torch.model_wrapper.superpoint.AutoImageProcessor")
    @patch("src.backend.torch.model_wrapper.superpoint.SuperPointForKeypointDetection")
    def test_load_uses_remote_checkpoint_when_local_empty(self, mock_model_class, mock_processor_class, wrapper,
                                                          mock_model, mock_processor, tmp_path):
        checkpoint = tmp_path / "empty"
        checkpoint.mkdir()
        wrapper._config = {"checkpoint": str(checkpoint)}

        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model

        wrapper.load()
        mock_processor_class.from_pretrained.assert_called_once_with("magic-leap-community/superpoint",
                                                                     local_files_only=False)

    @patch("src.backend.torch.model_wrapper.superpoint.AutoImageProcessor")
    @patch("src.backend.torch.model_wrapper.superpoint.SuperPointForKeypointDetection")
    def test_load_moves_model_to_device(self, mock_model_class, mock_processor_class, wrapper, mock_model,
                                        mock_processor, tmp_path):
        checkpoint = tmp_path / "superpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("{}")
        wrapper._config = {"checkpoint": str(checkpoint)}

        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        wrapper.load()
        mock_model.to.assert_called_once_with(wrapper._device)

    @patch("src.backend.torch.model_wrapper.superpoint.AutoImageProcessor")
    @patch("src.backend.torch.model_wrapper.superpoint.SuperPointForKeypointDetection")
    def test_load_logs_local_checkpoint_warning(self, mock_model_class, mock_processor_class, wrapper,
                                                mock_model, mock_processor, tmp_path):
        checkpoint = tmp_path / "missing"
        wrapper._config = {"checkpoint": str(checkpoint)}

        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model

        wrapper.load()

    @patch("src.backend.torch.model_wrapper.superpoint.AutoImageProcessor")
    @patch("src.backend.torch.model_wrapper.superpoint.SuperPointForKeypointDetection")
    def test_load_logs_remote_fallback(self, mock_model_class, mock_processor_class, wrapper, mock_model,
                                       mock_processor, tmp_path):
        checkpoint = tmp_path / "missing"
        wrapper._config = {"checkpoint": str(checkpoint)}

        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        wrapper.load()
        wrapper._logger.info.assert_any_call("Switching to remote repository: magic-leap-community/superpoint")

    @patch("src.backend.torch.model_wrapper.superpoint.AutoImageProcessor")
    @patch("src.backend.torch.model_wrapper.superpoint.SuperPointForKeypointDetection")
    def test_load_logs_error_on_failure(self, mock_model_class, mock_processor_class, wrapper, tmp_path):
        checkpoint = tmp_path / "superpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("{}")
        wrapper._config = {"checkpoint": str(checkpoint)}
        mock_processor_class.from_pretrained.side_effect = RuntimeError("checkpoint failed")

        result = wrapper.load()
        assert result is None
        wrapper._logger.error.assert_called_once_with(f"Failed to load from {checkpoint}: checkpoint failed")

    @patch("src.backend.torch.model_wrapper.superpoint.AutoImageProcessor")
    @patch("src.backend.torch.model_wrapper.superpoint.SuperPointForKeypointDetection")
    def test_load_uses_custom_local_files_only(self, mock_model_class, mock_processor_class, wrapper, mock_model,
                                               mock_processor, tmp_path):
        checkpoint = tmp_path / "superpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("{}")
        wrapper._config = {"checkpoint": str(checkpoint), "local_files_only": False}

        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        wrapper.load()
        mock_processor_class.from_pretrained.assert_called_once_with(str(checkpoint), local_files_only=False)


class TestSuperPointTorchModelWrapperCall:
    def test_call_returns_dict(self, wrapper, inference_outputs):
        wrapper._model.return_value = inference_outputs
        inputs = {"image": {"pixel_values": torch.rand(1, 3, 240, 320)}}

        result = wrapper.call(inputs)

        assert isinstance(result, dict)
        assert "inference_outputs" in result
        assert "image" in result

    def test_call_returns_model_outputs(self, wrapper, inference_outputs):
        wrapper._model.return_value = inference_outputs
        inputs = {"image": {"pixel_values": torch.rand(1, 3, 240, 320)}}

        result = wrapper.call(inputs)

        assert result["inference_outputs"] is inference_outputs

    def test_call_invokes_model(self, wrapper, inference_outputs):
        wrapper._model.return_value = inference_outputs
        image = {"pixel_values": torch.rand(1, 3, 240, 320)}

        wrapper.call({"image": image})

        wrapper._model.assert_called_once_with(**image)

    def test_call_preserves_input_fields(self, wrapper, inference_outputs):
        wrapper._model.return_value = inference_outputs
        image = {"pixel_values": torch.rand(1, 3, 240, 320)}
        inputs = {
            "image": image,
            "height": 240,
            "width": 320,
        }

        result = wrapper.call(inputs)
        assert result["image"] is image
        assert result["height"] == 240
        assert result["width"] == 320

    def test_call_logs_error_on_failure(self, wrapper):
        wrapper._model.side_effect = RuntimeError("inference failed")
        image = {"pixel_values": torch.rand(1, 3, 240, 320)}

        with pytest.raises(RuntimeError, match="inference failed"):
            wrapper.call({"image": image})

        wrapper._logger.error.assert_called_once_with("Super Point inference error: inference failed")

    def test_call_reraises_model_error(self, wrapper):
        wrapper._model.side_effect = ValueError("invalid input")
        image = {"pixel_values": torch.rand(1, 3, 240, 320)}

        with pytest.raises(ValueError, match="invalid input"):
            wrapper.call({"image": image})
