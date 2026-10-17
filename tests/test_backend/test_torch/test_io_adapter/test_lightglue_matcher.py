from unittest.mock import MagicMock, patch

import pytest
import torch

from src.backend.torch.model_wrapper.lightglue_matcher import LightGlueTorchModelWrapper


@pytest.fixture
def wrapper():
    obj = LightGlueTorchModelWrapper.__new__(LightGlueTorchModelWrapper)
    obj._device = torch.device("cpu")
    obj._logger = MagicMock()
    obj._config = {}
    obj._model = None
    return obj


@pytest.fixture
def descriptor_class():
    descriptor = MagicMock()
    descriptor._descriptor_name = "superpoint_lightglue"
    return descriptor


class TestLightGlueTorchModelWrapperLoad:
    @patch(f"src.backend.torch.model_wrapper.lightglue_matcher.LightGlue")
    def test_load_creates_model(self, mock_lightglue, wrapper, descriptor_class):
        model = MagicMock()
        mock_lightglue.return_value.eval.return_value.to.return_value = model

        wrapper._config = {"descriptor_class": descriptor_class}

        result = wrapper.load()
        mock_lightglue.assert_called_once_with(features="superpoint", descriptor_class=descriptor_class)
        mock_lightglue.return_value.eval.assert_called_once_with()
        mock_lightglue.return_value.eval.return_value.to.assert_called_once_with(wrapper._device)

        assert result == {"model": model}
        assert wrapper._model is model

    @patch(f"src.backend.torch.model_wrapper.lightglue_matcher.LightGlue")
    def test_load_uses_descriptor_name_without_suffix(self, mock_lightglue, wrapper, descriptor_class):
        wrapper._config = {"descriptor_class": descriptor_class}
        mock_lightglue.return_value.eval.return_value.to.return_value = (MagicMock())

        wrapper.load()
        assert (mock_lightglue.call_args.kwargs["features"] == "superpoint")

    @patch(f"src.backend.torch.model_wrapper.lightglue_matcher.LightGlue")
    def test_load_passes_additional_configuration(self, mock_lightglue, wrapper, descriptor_class):
        wrapper._config = {
            "descriptor_class": descriptor_class,
            "depth_confidence": 0.95,
            "width_confidence": 0.99,
        }
        mock_lightglue.return_value.eval.return_value.to.return_value = (MagicMock())
        wrapper.load()

        kwargs = mock_lightglue.call_args.kwargs
        assert kwargs["depth_confidence"] == 0.95
        assert kwargs["width_confidence"] == 0.99

    @patch(f"src.backend.torch.model_wrapper.lightglue_matcher.LightGlue")
    def test_load_logs_device(self, mock_lightglue, wrapper, descriptor_class):
        wrapper._config = {"descriptor_class": descriptor_class}
        mock_lightglue.return_value.eval.return_value.to.return_value = (MagicMock())

        wrapper.load()
        wrapper._logger.info.assert_called_once()


class TestLightGlueTorchModelWrapperCall:
    def test_call_returns_matches_and_scores(self, wrapper):
        matches = torch.tensor([[[0, 0], [1, 1]]])
        scores = torch.tensor([[0.95, 0.85]])

        model = MagicMock()
        model.return_value = {
            "matches": matches,
            "scores": scores,
        }
        wrapper._model = model

        result = wrapper.call({
            "image0": {"keypoints": torch.rand(1, 3, 2)},
            "image1": {"keypoints": torch.rand(1, 2, 2)},
        })

        assert result["matches"] is matches
        assert result["scores"] is scores

    def test_call_passes_pipeline_features_to_model(self, wrapper):
        features0 = {
            "keypoints": torch.rand(1, 3, 2),
            "descriptors": torch.rand(1, 3, 256),
            "keypoint_scores": torch.rand(1, 3),
        }
        features1 = {
            "keypoints": torch.rand(1, 2, 2),
            "descriptors": torch.rand(1, 2, 256),
            "keypoint_scores": torch.rand(1, 2),
        }
        inputs = {"image0": features0, "image1": features1}

        model = MagicMock(return_value={
            "matches": torch.empty((1, 0, 2), dtype=torch.long),
            "scores": torch.empty((1, 0)),
        })
        wrapper._model = model

        wrapper.call(inputs)
        model.assert_called_once_with(inputs)

    def test_call_runs_without_gradients(self, wrapper):
        model = MagicMock(return_value={
            "matches": torch.empty((1, 0, 2), dtype=torch.long),
            "scores": torch.empty((1, 0)),
        })
        wrapper._model = model

        with patch(f"src.backend.torch.model_wrapper.lightglue_matcher.torch.no_grad") as mock_no_grad:
            mock_no_grad.return_value.__enter__ = MagicMock()
            mock_no_grad.return_value.__exit__ = MagicMock(return_value=False)

            wrapper.call({"image0": {}, "image1": {}})
            mock_no_grad.assert_called_once_with()

    def test_call_supports_empty_matches(self, wrapper):
        matches = torch.empty((1, 0, 2), dtype=torch.long)
        scores = torch.empty((1, 0))

        wrapper._model = MagicMock(return_value={
            "matches": matches,
            "scores": scores,
        })

        result = wrapper.call({"image0": {}, "image1": {}})

        assert result["matches"].shape == (1, 0, 2)
        assert result["scores"].shape == (1, 0)

    def test_call_propagates_model_error_and_logs(self, wrapper):
        wrapper._model = MagicMock(side_effect=RuntimeError("inference failed"))

        with pytest.raises(RuntimeError, match="inference failed"):
            wrapper.call({"image0": {}, "image1": {}})

        wrapper._logger.error.assert_called_once()
        assert "LightGlue inference error" in (wrapper._logger.error.call_args.args[0])

    def test_call_raises_if_model_is_not_loaded(self, wrapper):
        wrapper._model = None

        with pytest.raises((TypeError, AttributeError)):
            wrapper.call({"image0": {}, "image1": {}})
