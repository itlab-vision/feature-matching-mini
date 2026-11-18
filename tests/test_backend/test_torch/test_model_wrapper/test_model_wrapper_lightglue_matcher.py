import numpy as np
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
        "keypoints": np.random.rand(10, 2),
        "descriptors": np.random.rand(10, 256),
        "scores": np.random.rand(10),
        "image_size": np.array([[100, 100]]),
    }


@pytest.fixture
def features1():
    return {
        "keypoints": np.random.rand(8, 2),
        "descriptors": np.random.rand(8, 256),
        "scores": np.random.rand(8),
        "image_size": np.array([[100, 100]]),
    }


@pytest.fixture
def matcher_inputs(features0, features1):
    return {"features0": features0, "features1": features1}


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
    @pytest.mark.parametrize("count", [0, 2])
    def test_call_converts_pair_input_and_output(self, wrapper, matcher_inputs, count):
        matches = torch.arange(count * 2).reshape(1, count, 2)
        scores = torch.ones(1, count)

        def infer(data):
            assert not torch.is_grad_enabled()
            for i in range(2):
                assert data[f"image{i}"]["keypoints"].shape == (1, 10 - 2 * i, 2)
            return {"matches": matches, "scores": scores}

        wrapper._model = MagicMock(side_effect=infer)
        result = wrapper.call(matcher_inputs)
        np.testing.assert_array_equal(result["matches"], matches.numpy()[0])
        np.testing.assert_array_equal(result["scores"], scores.numpy()[0])

    def test_call_logs_and_reraises_model_error(self, wrapper, matcher_inputs):
        wrapper._model = MagicMock(side_effect=RuntimeError("boom"))
        with pytest.raises(RuntimeError, match="boom"):
            wrapper.call(matcher_inputs)
        wrapper._logger.error.assert_called_once_with("LightGlue inference error: boom")
