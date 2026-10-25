import pytest
import torch

from unittest.mock import MagicMock, patch
from logging import Logger

from src.backend.torch.model_wrapper.lightglue_matcher import LightGlueTorchModelWrapper


@pytest.fixture
def mock_logger():
    return MagicMock(spec=Logger)


@pytest.fixture
def descriptor_class():
    descriptor = MagicMock()
    descriptor._descriptor_name = "superpoint_lightglue"
    return descriptor


@pytest.fixture
def wrapper(mock_logger, descriptor_class):
    obj = LightGlueTorchModelWrapper.__new__(LightGlueTorchModelWrapper)
    obj._logger = mock_logger
    obj._device = torch.device("cpu")
    obj._model_name = "lightglue"
    obj._config = {"descriptor_class": descriptor_class}
    obj._model = None
    return obj


@pytest.fixture
def features0():
    return {
        "keypoints": torch.rand(10, 2),
        "descriptors": torch.rand(10, 256),
        "scores": torch.rand(10),
        "image_size": torch.tensor([100, 100]),
    }


@pytest.fixture
def features1():
    return {
        "keypoints": torch.rand(8, 2),
        "descriptors": torch.rand(8, 256),
        "scores": torch.rand(8),
        "image_size": torch.tensor([100, 100]),
    }


@pytest.fixture
def matcher_inputs(features0, features1):
    return {
        "image0": {
            "keypoints": features0["keypoints"].unsqueeze(0),
            "descriptors": features0["descriptors"].unsqueeze(0),
            "keypoint_scores": features0["scores"].unsqueeze(0),
            "image_size": features0["image_size"],
        },
        "image1": {
            "keypoints": features1["keypoints"].unsqueeze(0),
            "descriptors": features1["descriptors"].unsqueeze(0),
            "keypoint_scores": features1["scores"].unsqueeze(0),
            "image_size": features1["image_size"],
        },
    }


class TestLightGlueTorchModelWrapperLoad:
    @patch("src.backend.torch.model_wrapper.lightglue_matcher.LightGlue")
    def test_load(self, mock_lightglue, wrapper):
        mock_model = MagicMock()
        mock_lightglue.return_value.eval.return_value.to.return_value = (mock_model)
        result = wrapper.load()

        mock_lightglue.assert_called_once_with(features="superpoint",
                                               descriptor_class=wrapper._config["descriptor_class"])
        assert result["model"] is mock_model
        assert wrapper._model is mock_model

    @pytest.mark.parametrize(
        "descriptor_name,expected",
        [
            ("superpoint_lightglue", "superpoint"),
            ("disk_lightglue", "disk"),
            ("sift_lightglue", "sift"),
            ("aliked_lightglue", "aliked"),
            ("doghardnet_lightglue", "doghardnet"),
        ],
    )
    @patch("src.backend.torch.model_wrapper.lightglue_matcher.LightGlue")
    def test_load_selects_extractor(self, mock_lightglue, wrapper, descriptor_class, descriptor_name, expected):
        descriptor_class._descriptor_name = descriptor_name
        mock_lightglue.return_value.eval.return_value.to.return_value = (MagicMock())

        wrapper.load()
        assert mock_lightglue.call_args.kwargs["features"] == expected

    @patch("src.backend.torch.model_wrapper.lightglue_matcher.LightGlue")
    def test_load_sets_eval_and_device(self, mock_lightglue, wrapper):
        model_instance = mock_lightglue.return_value
        evaluated_model = model_instance.eval.return_value
        final_model = evaluated_model.to.return_value

        result = wrapper.load()
        model_instance.eval.assert_called_once_with()
        evaluated_model.to.assert_called_once_with(torch.device("cpu"))
        assert result["model"] is final_model

    @patch("src.backend.torch.model_wrapper.lightglue_matcher.LightGlue")
    def test_load_logs_initialization(self, mock_lightglue, wrapper):
        mock_lightglue.return_value.eval.return_value.to.return_value = (MagicMock())

        wrapper.load()
        wrapper._logger.info.assert_called()


class TestLightGlueTorchModelWrapperCall:
    def test_call_returns_matches_and_scores(self, wrapper, matcher_inputs):
        matches = torch.tensor([[[0, 0], [1, 1]]], dtype=torch.long)
        scores = torch.tensor([[0.95, 0.85]])

        mock_model = MagicMock()
        mock_model.return_value = {
            "matches": matches,
            "scores": scores,
        }
        wrapper._model = mock_model
        result = wrapper.call(matcher_inputs)

        assert set(result.keys()) == {"matches", "scores"}
        assert result["matches"] is matches
        assert result["scores"] is scores

    def test_call_invokes_model_with_pipeline_output(self, wrapper, matcher_inputs):
        mock_model = MagicMock(
            return_value={
                "matches": torch.empty(1, 0, 2, dtype=torch.long),
                "scores": torch.empty(1, 0),
            }
        )
        wrapper._model = mock_model
        wrapper.call(matcher_inputs)
        mock_model.assert_called_once_with(matcher_inputs)

    def test_call_runs_without_gradients(self, wrapper, matcher_inputs):
        mock_model = MagicMock(
            return_value={
                "matches": torch.empty(1, 0, 2, dtype=torch.long),
                "scores": torch.empty(1, 0),
            }
        )
        wrapper._model = mock_model

        with patch("src.backend.torch.model_wrapper.lightglue_matcher.torch.no_grad") as mock_no_grad:
            mock_no_grad.return_value.__enter__.return_value = None
            mock_no_grad.return_value.__exit__.return_value = False
            wrapper.call(matcher_inputs)

        mock_no_grad.assert_called_once_with()

    def test_call_supports_empty_matches(self, wrapper, matcher_inputs):
        mock_model = MagicMock(
            return_value={
                "matches": torch.empty(1, 0, 2, dtype=torch.long),
                "scores": torch.empty(1, 0),
            }
        )
        wrapper._model = mock_model
        result = wrapper.call(matcher_inputs)

        assert result["matches"].shape == (1, 0, 2)
        assert result["scores"].shape == (1, 0)

    def test_call_logs_and_reraises_model_error(self, wrapper, matcher_inputs):
        error = RuntimeError("inference failed")
        mock_model = MagicMock(side_effect=error)
        wrapper._model = mock_model

        with pytest.raises(RuntimeError, match="inference failed"):
            wrapper.call(matcher_inputs)

        wrapper._logger.error.assert_called_once()
        assert "LightGlue inference error" in (wrapper._logger.error.call_args.args[0])

    def test_call_propagates_missing_model_error(self, wrapper, matcher_inputs):
        wrapper._model = None

        with pytest.raises(TypeError):
            wrapper.call(matcher_inputs)

        wrapper._logger.error.assert_called_once()


class TestLightGlueTorchModelWrapperIntegration:
    @patch("src.backend.torch.model_wrapper.lightglue_matcher.LightGlue")
    def test_load_then_call(self, mock_lightglue, wrapper, matcher_inputs):
        matches = torch.tensor([[[0, 0], [1, 1]]], dtype=torch.long)
        scores = torch.tensor([[0.9, 0.8]])

        mock_model = MagicMock()
        mock_model.return_value = {
            "matches": matches,
            "scores": scores,
        }
        mock_lightglue.return_value.eval.return_value.to.return_value = (mock_model)
        loaded = wrapper.load()
        result = wrapper.call(matcher_inputs)

        assert loaded["model"] is mock_model
        assert torch.equal(result["matches"], matches)
        assert torch.equal(result["scores"], scores)
