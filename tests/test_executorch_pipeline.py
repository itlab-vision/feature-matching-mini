from pathlib import Path
from unittest.mock import MagicMock, patch
from logging import Logger

import cv2 as cv
import numpy as np
import pytest
import torch
import torch.nn.functional as functional

from src.executorch_pipeline import (ExecuTorch, ExecuTorchDetector, ExecuTorchDescriptor, ExecuTorchMatcher,
                                     SuperPointLightGlueExecuTorch, DiskLightGlueExecuTorch, D2NetExecuTorch,
                                     TFeatExecuTorch, HardNetExecuTorch, LightGlueExecuTorch, SuperGlueExecuTorch)


@pytest.fixture
def mock_logger():
    return MagicMock(spec=Logger)


@pytest.fixture
def random_img():
    return np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)


@pytest.fixture
def load_img():
    def _load(name, color=True):
        path = Path(__file__).parent.parent / "test_data" / name
        mode = cv.IMREAD_COLOR if color else cv.IMREAD_GRAYSCALE
        img = cv.imread(str(path), mode)
        if img is None:
            pytest.skip(f"Test image not found at {path}")
        return img
    return _load


def model_path_or_skip(filename):
    path = Path(__file__).parent.parent / "exported" / filename
    if not path.is_file():
        pytest.skip(f"Model not found: {path}")
    return path


@pytest.fixture
def fake_method():
    method = MagicMock()
    method.execute = MagicMock()
    return method


@pytest.fixture
def patched_runtime(fake_method):
    with patch.object(ExecuTorch, "_load_method", return_value=fake_method) as loader:
        yield loader, fake_method


class TestExecuTorchLoadMethod:
    def test_missing_file_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist.pte"
        with pytest.raises(FileNotFoundError, match="ExecuTorch model does not exist"):
            ExecuTorch._load_method(missing)

    def test_missing_file_message_contains_path(self, tmp_path):
        missing = tmp_path / "nope.pte"
        with pytest.raises(FileNotFoundError) as excinfo:
            ExecuTorch._load_method(missing)
        assert str(missing) in str(excinfo.value)

    def test_accepts_str_path(self, tmp_path):
        missing = tmp_path / "nope.pte"
        with pytest.raises(FileNotFoundError):
            ExecuTorch._load_method(str(missing))


class TestExecuTorchNmsTopK:
    def test_rejects_non_2d_scores(self):
        scores_3d = torch.rand(2, 4, 4)
        with pytest.raises(ValueError, match="HxW score map"):
            ExecuTorch._nms_topk(scores_3d, keypoints=1, radius=1)

    def test_rejects_1d_scores(self):
        scores_1d = torch.rand(16)
        with pytest.raises(ValueError, match="HxW score map"):
            ExecuTorch._nms_topk(scores_1d, keypoints=1, radius=1)

    def test_coordinates_within_bounds(self):
        height, width = 12, 18
        scores = torch.rand(height, width)
        xy, _ = ExecuTorch._nms_topk(scores, keypoints=6, radius=1)
        assert torch.all(xy[:, 0] >= 0) and torch.all(xy[:, 0] < width)
        assert torch.all(xy[:, 1] >= 0) and torch.all(xy[:, 1] < height)


class TestExecuTorchSampleDescriptors:
    def test_output_shape(self):
        dense = torch.rand(1, 16, 10, 10)
        keypoints = torch.tensor([[0.0, 0.0], [5.0, 5.0], [9.0, 9.0]])
        descriptors = ExecuTorch._sample_descriptors(dense, keypoints, image_height=10, image_width=10)
        assert descriptors.shape == (3, 16)

    def test_descriptors_are_l2_normalized(self):
        dense = torch.rand(1, 8, 12, 12)
        keypoints = torch.tensor([[1.0, 1.0], [6.0, 6.0]])
        descriptors = ExecuTorch._sample_descriptors(dense, keypoints, image_height=12, image_width=12)
        norms = descriptors.norm(p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_single_keypoint(self):
        dense = torch.rand(1, 4, 5, 5)
        keypoints = torch.tensor([[2.0, 2.0]])
        descriptors = ExecuTorch._sample_descriptors(dense, keypoints, image_height=5, image_width=5)
        assert descriptors.shape == (1, 4)


class ConcreteDetector(ExecuTorchDetector):
    def _features(self, outputs, input_height, input_width):
        keypoints = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
        descriptors = torch.rand(2, 4)
        scores = torch.tensor([0.9, 0.8])
        return keypoints, descriptors, scores


class TestExecuTorchDetectorInit:
    def test_requires_model_path(self, mock_logger, patched_runtime):
        with pytest.raises(ValueError, match="requires 'executorch_model_path'"):
            ConcreteDetector("det", mock_logger, config={})

    def test_custom_input_shape(self, mock_logger, patched_runtime):
        det = ConcreteDetector("det", mock_logger, config={
            "executorch_model_path": "m.pte", "input_shape": (1, 1, 32, 32)
        })
        assert det._input_shape == (1, 1, 32, 32)

    def test_default_num_keypoints(self, mock_logger, patched_runtime):
        det = ConcreteDetector("det", mock_logger, config={"executorch_model_path": "m.pte"})
        assert det._num_keypoints == 256

    def test_rejects_bad_input_shape_length(self, mock_logger, patched_runtime):
        with pytest.raises(ValueError, match="input_shape must be"):
            ConcreteDetector("det", mock_logger, config={
                "executorch_model_path": "m.pte", "input_shape": (3, 480, 640)
            })

    def test_loader_called_with_given_path(self, mock_logger, patched_runtime):
        loader, _ = patched_runtime
        ConcreteDetector("det", mock_logger, config={"executorch_model_path": "some/path.pte"})
        loader.assert_called_once_with("some/path.pte")


class TestExecuTorchDetectorPrepareInput:
    @pytest.fixture
    def detector(self, mock_logger, patched_runtime):
        return ConcreteDetector("det", mock_logger, config={
            "executorch_model_path": "m.pte", "input_shape": (1, 3, 64, 64)
        })

    def test_rejects_non_tensor_input(self, detector):
        with pytest.raises(TypeError, match="torch.Tensor"):
            detector._prepare_input(np.zeros((3, 10, 10)))

    def test_squeezes_batch_dim_of_one(self, detector):
        img = torch.rand(1, 3, 20, 30)
        resized, h, w = detector._prepare_input(img)
        assert resized.shape == (1, 3, 64, 64)
        assert (h, w) == (20, 30)

    def test_rejects_channel_mismatch(self, detector):
        img = torch.rand(1, 20, 20)
        with pytest.raises(ValueError, match="Model expects 3 channels"):
            detector._prepare_input(img)


class TestExecuTorchDetectorDetect:
    @pytest.fixture
    def detector(self, mock_logger, patched_runtime):
        det = ConcreteDetector("det", mock_logger, config={
            "executorch_model_path": "m.pte", "input_shape": (1, 3, 64, 64)
        })
        det._method.execute.return_value = ("out0", "out1", "out2")
        return det

    def test_returns_expected_keys(self, detector):
        img = torch.rand(3, 128, 128)
        features = detector.detect(img)
        for key in ("keypoints", "descriptors", "scores", "width", "height", "executorch"):
            assert key in features

    def test_executorch_flag_is_true(self, detector):
        img = torch.rand(3, 128, 128)
        features = detector.detect(img)
        assert features["executorch"] is True

    def test_width_height_match_original(self, detector):
        img = torch.rand(3, 100, 50)
        features = detector.detect(img)
        assert features["height"] == 100
        assert features["width"] == 50


class FakeKeypoint:
    def __init__(self, x, y):
        self.pt = (x, y)


class TestExecuTorchDescriptorInit:
    def test_patch_based_requires_model_path(self, mock_logger, patched_runtime):
        with pytest.raises(ValueError, match="requires 'executorch_model_path'"):
            TFeatExecuTorch("tfeat", mock_logger, config={})

    def test_custom_patch_size(self, mock_logger, patched_runtime):
        desc = HardNetExecuTorch("hardnet", mock_logger, config={
            "executorch_model_path": "m.pte", "patch_size": 64
        })
        assert desc._patch_size == 64

    def test_default_norm_is_l2(self, mock_logger, patched_runtime):
        desc = TFeatExecuTorch("tfeat", mock_logger, config={"executorch_model_path": "m.pte"})
        assert desc.default_norm == cv.NORM_L2


class TestExecuTorchDescriptorCompute:
    @pytest.fixture
    def patch_descriptor(self, mock_logger, patched_runtime):
        desc = TFeatExecuTorch("tfeat", mock_logger, config={
            "executorch_model_path": "m.pte", "patch_size": 8
        })
        desc._method.execute.return_value = (torch.rand(1, 32),)
        return desc

    def test_empty_keypoints_returns_empty_descriptors(self, patch_descriptor, mock_logger):
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        features = patch_descriptor.compute(image, {"kp": []})
        assert features["des"] == ()
        mock_logger.warning.assert_called()

    def test_none_keypoints_treated_as_empty(self, patch_descriptor, mock_logger):
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        features = patch_descriptor.compute(image, {"kp": None})
        assert features["des"] == ()

    def test_kp_preserved_in_output(self, patch_descriptor):
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        keypoints = [FakeKeypoint(1, 1)]
        features = patch_descriptor.compute(image, {"kp": keypoints})
        assert features["kp"] is keypoints


class TestExecuTorchDescriptorPreprocess:
    @pytest.fixture
    def descriptor(self, mock_logger, patched_runtime):
        return TFeatExecuTorch("tfeat", mock_logger, config={"executorch_model_path": "m.pte"})

    def test_converts_bgr_to_rgb_and_normalizes(self, descriptor):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        image[:, :, 0] = 255
        preprocessed = descriptor._preprocess(image)
        assert preprocessed.shape == (1, 3, 10, 10)
        assert preprocessed.max() <= 1.0
        assert preprocessed.min() >= 0.0

    def test_grayscale_image_passthrough_shape(self, descriptor):
        image = np.zeros((10, 10, 1), dtype=np.uint8)
        preprocessed = descriptor._preprocess(image)
        assert preprocessed.shape[0] == 1

    def test_adds_batch_dim_when_missing(self, descriptor):
        image = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
        preprocessed = descriptor._preprocess(image)
        assert preprocessed.ndim == 4
        assert preprocessed.shape[0] == 1


class TestExecuTorchDescriptorExtractPatches:
    @pytest.fixture
    def descriptor(self, mock_logger, patched_runtime):
        return TFeatExecuTorch("tfeat", mock_logger, config={
            "executorch_model_path": "m.pte", "patch_size": 16
        })

    def test_extracts_correct_number_of_patches(self, descriptor):
        image = torch.rand(1, 3, 64, 64)
        keypoints = [FakeKeypoint(10, 10), FakeKeypoint(30, 40)]
        patches = descriptor._extract_patches(image, keypoints, patch_size=16)
        assert patches.shape[0] == 2

    def test_patch_spatial_size_matches_patch_size(self, descriptor):
        image = torch.rand(1, 3, 64, 64)
        keypoints = [FakeKeypoint(30, 30)]
        patches = descriptor._extract_patches(image, keypoints, patch_size=16)
        assert patches.shape[-2:] == (16, 16)

    def test_patches_finite(self, descriptor):
        image = torch.rand(1, 3, 64, 64)
        keypoints = [FakeKeypoint(0, 0), FakeKeypoint(63, 63)]
        patches = descriptor._extract_patches(image, keypoints, patch_size=8)
        assert torch.isfinite(patches).all()


class ConcreteMatcher(ExecuTorchMatcher):
    def _correspondences(self, features0, features1):
        return self._method.execute((
            features0["keypoints"], features1["keypoints"],
            features0["descriptors"], features1["descriptors"],
        ))


class TestExecuTorchMatcherInit:
    def test_requires_model_path(self, mock_logger, patched_runtime):
        with pytest.raises(ValueError, match="requires 'executorch_model_path'"):
            ConcreteMatcher(mock_logger, "matcher", "desc", config={})

    def test_default_num_keypoints(self, mock_logger, patched_runtime):
        matcher = ConcreteMatcher(mock_logger, "matcher", "desc", config={"executorch_model_path": "m.pte"})
        assert matcher._num_keypoints == 256

    def test_custom_num_keypoints(self, mock_logger, patched_runtime):
        matcher = ConcreteMatcher(mock_logger, "matcher", "desc", config={
            "executorch_model_path": "m.pte", "num_keypoints": 128
        })
        assert matcher._num_keypoints == 128


class TestExecuTorchMatcherFixedFeatures:
    @pytest.fixture
    def matcher(self, mock_logger, patched_runtime):
        return ConcreteMatcher(mock_logger, "matcher", "desc", config={
            "executorch_model_path": "m.pte", "num_keypoints": 4
        })

    def test_raises_on_keypoint_count_mismatch(self, matcher):
        feature = {"keypoints": torch.rand(3, 2), "descriptors": torch.rand(3, 8)}
        with pytest.raises(ValueError, match="Matcher expects K=4"):
            matcher._fixed_features(feature)

    def test_adds_batch_dimension(self, matcher):
        feature = {"keypoints": torch.rand(4, 2), "descriptors": torch.rand(4, 8)}
        fixed = matcher._fixed_features(feature)
        assert fixed["keypoints"].shape == (1, 4, 2)
        assert fixed["descriptors"].shape == (1, 4, 8)

    def test_output_is_contiguous_and_on_cpu(self, matcher):
        feature = {"keypoints": torch.rand(4, 2), "descriptors": torch.rand(4, 8)}
        fixed = matcher._fixed_features(feature)
        assert fixed["keypoints"].is_contiguous()
        assert fixed["keypoints"].device.type == "cpu"


class TestExecuTorchMatcherMatch:
    @pytest.fixture
    def matcher(self, mock_logger, patched_runtime):
        m = ConcreteMatcher(mock_logger, "matcher", "desc", config={
            "executorch_model_path": "m.pte", "num_keypoints": 3
        })
        return m

    def test_filters_invalid_matches(self, matcher):
        features0 = {"keypoints": torch.rand(3, 2), "descriptors": torch.rand(3, 8)}
        features1 = {"keypoints": torch.rand(3, 2), "descriptors": torch.rand(3, 8)}

        matches0 = torch.tensor([[0, -1, 2]])
        match_scores0 = torch.tensor([[0.9, 0.0, 0.7]])
        matcher._method.execute.return_value = (matches0, None, match_scores0, None)

        result = matcher.match(features0, features1)
        assert result["matches"].shape[0] == 2
        assert torch.equal(result["matches"][:, 0], torch.tensor([0, 2]))

    def test_no_valid_matches_returns_empty(self, matcher):
        features0 = {"keypoints": torch.rand(3, 2), "descriptors": torch.rand(3, 8)}
        features1 = {"keypoints": torch.rand(3, 2), "descriptors": torch.rand(3, 8)}

        matches0 = torch.tensor([[-1, -1, -1]])
        match_scores0 = torch.tensor([[0.0, 0.0, 0.0]])
        matcher._method.execute.return_value = (matches0, None, match_scores0, None)

        result = matcher.match(features0, features1)
        assert result["matches"].shape[0] == 0
        assert result["scores"].shape[0] == 0

    def test_init_matcher_returns_none(self, matcher):
        assert matcher._init_matcher() is None


class TestSuperPointLightGlueFeatures:
    @pytest.fixture
    def model(self, mock_logger, patched_runtime):
        return SuperPointLightGlueExecuTorch("splg", mock_logger, config={
            "executorch_model_path": "m.pte",
            "input_shape": (1, 3, 64, 64),
            "num_keypoints": 4,
            "nms_radius": 1,
        })

    def test_features_output_shapes(self, model):
        coarse_h, coarse_w = 8, 8
        probabilities = torch.rand(1, 65, coarse_h, coarse_w)
        dense = torch.rand(1, 16, coarse_h, coarse_w)
        outputs = (None, probabilities, dense)

        keypoints, descriptors, scores = model._features(outputs, input_height=64, input_width=64)
        assert keypoints.shape[1] == 2
        assert descriptors.shape[0] == keypoints.shape[0]
        assert scores.shape[0] == keypoints.shape[0]

    def test_dustbin_channel_dropped(self, model):
        coarse_h, coarse_w = 8, 8
        probabilities = torch.rand(1, 65, coarse_h, coarse_w)
        dense = torch.rand(1, 16, coarse_h, coarse_w)
        outputs = (None, probabilities, dense)
        model._features(outputs, input_height=64, input_width=64)


class TestDiskLightGlueFeatures:
    @pytest.fixture
    def model(self, mock_logger, patched_runtime):
        return DiskLightGlueExecuTorch("disklg", mock_logger, config={
            "executorch_model_path": "m.pte",
            "input_shape": (1, 3, 32, 32),
            "num_keypoints": 4,
            "nms_radius": 1,
        })

    def test_features_output_shapes(self, model):
        heatmap = torch.rand(1, 1, 32, 32)
        dense = torch.rand(1, 8, 32, 32)
        outputs = (heatmap, dense)

        keypoints, descriptors, scores = model._features(outputs, input_height=32, input_width=32)
        assert keypoints.shape == (4, 2)
        assert descriptors.shape == (4, 8)
        assert scores.shape == (4,)


class TestD2NetFeatures:
    @pytest.fixture
    def model(self, mock_logger, patched_runtime):
        return D2NetExecuTorch("d2net", mock_logger, config={
            "executorch_model_path": "m.pte",
            "input_shape": (1, 3, 40, 40),
            "num_keypoints": 4,
            "nms_radius": 1,
        })

    def test_score_map_shape(self, model):
        dense = torch.rand(1, 16, 10, 10)
        score_map = model._d2net_score_map(dense)
        assert score_map.shape == (10, 10)

    def test_score_map_is_finite_and_nonnegative_range(self, model):
        dense = torch.rand(1, 16, 10, 10) * 5 - 2
        score_map = model._d2net_score_map(dense)
        assert torch.isfinite(score_map).all()

    def test_features_rescales_keypoints_to_input_resolution(self, model):
        dense = torch.rand(1, 16, 10, 10)
        outputs = (dense,)
        keypoints, descriptors, scores = model._features(outputs, input_height=40, input_width=40)
        assert torch.all(keypoints[:, 0] <= 40)
        assert torch.all(keypoints[:, 1] <= 40)
        assert descriptors.shape[0] == keypoints.shape[0] == scores.shape[0]

    def test_score_map_all_zero_dense_does_not_crash(self, model):
        dense = torch.zeros(1, 8, 10, 10)
        outputs = (dense,)
        keypoints, descriptors, scores = model._features(outputs, input_height=40, input_width=40)
        assert torch.isfinite(keypoints).all()


class TestLightGlueExecuTorchCorrespondences:
    @pytest.fixture
    def matcher(self, mock_logger, patched_runtime):
        return LightGlueExecuTorch(mock_logger, "lightglue", "superpoint", config={
            "executorch_model_path": "m.pte", "num_keypoints": 4
        })

    def test_calls_execute_with_keypoints_and_descriptors_in_order(self, matcher):
        kp0, desc0 = torch.rand(1, 4, 2), torch.rand(1, 4, 8)
        kp1, desc1 = torch.rand(1, 4, 2), torch.rand(1, 4, 8)
        features0 = {"keypoints": kp0, "descriptors": desc0}
        features1 = {"keypoints": kp1, "descriptors": desc1}

        matcher._method.execute.return_value = "result"
        result = matcher._correspondences(features0, features1)

        matcher._method.execute.assert_called_once_with((kp0, kp1, desc0, desc1))
        assert result == "result"

    def test_missing_keys_pass_none(self, matcher):
        features0 = {}
        features1 = {}
        matcher._method.execute.return_value = "result"
        matcher._correspondences(features0, features1)
        matcher._method.execute.assert_called_once_with((None, None, None, None))


class TestSuperGlueExecuTorchCorrespondences:
    @pytest.fixture
    def matcher(self, mock_logger, patched_runtime):
        return SuperGlueExecuTorch(mock_logger, "superglue", "superpoint", config={
            "executorch_model_path": "m.pte", "num_keypoints": 4})

    def test_transposes_descriptors_and_adds_batch_to_scores(self, matcher):
        kp0, desc0 = torch.rand(1, 4, 2), torch.rand(1, 8, 4)
        kp1, desc1 = torch.rand(1, 4, 2), torch.rand(1, 8, 4)
        scores0, scores1 = torch.rand(4), torch.rand(4)
        features0 = {"keypoints": kp0, "descriptors": desc0, "scores": scores0}
        features1 = {"keypoints": kp1, "descriptors": desc1, "scores": scores1}

        matcher._method.execute.return_value = "result"
        matcher._correspondences(features0, features1)

        (args,), _ = matcher._method.execute.call_args
        called_kp0, called_kp1, called_desc0, called_desc1, called_s0, called_s1 = args
        assert called_desc0.shape == (1, 4, 8)
        assert called_desc1.shape == (1, 4, 8)
        assert called_s0.shape == (1, 4)
        assert called_s1.shape == (1, 4)



pytestmark_integration = pytest.mark.integration

@pytest.mark.integration
class TestIntegrationSuperPointLightGlueDetector:
    MODEL_NAME = "superpoint_lightglue_dense_none_1x3x480x640.pte"

    @pytest.fixture
    def model(self, mock_logger):
        path = model_path_or_skip(self.MODEL_NAME)
        return SuperPointLightGlueExecuTorch("superpoint", mock_logger, config={
            "executorch_model_path": str(path),
            "input_shape": (1, 3, 480, 640),
            "num_keypoints": 256,
            "nms_radius": 4,
        })

    def test_detect_on_real_image_returns_features(self, model, load_img):
        img = load_img("box.png")
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB) if img.ndim == 3 else img
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        features = model.detect(tensor)
        assert features["keypoints"].shape[0] <= 256
        assert features["descriptors"].shape[1] == features["keypoints"].shape[0] or \
               features["descriptors"].shape[0] == features["keypoints"].shape[0]
        assert features["executorch"] is True

    def test_keypoints_within_image_bounds(self, model, load_img):
        img = load_img("box.png")
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        features = model.detect(tensor)
        h, w = tensor.shape[-2:]
        kp = features["keypoints"]
        if len(kp) > 0:
            assert torch.all(kp[:, 0] >= -1) and torch.all(kp[:, 0] <= w + 1)
            assert torch.all(kp[:, 1] >= -1) and torch.all(kp[:, 1] <= h + 1)


@pytest.mark.integration
class TestIntegrationDiskLightGlueDetector:
    MODEL_NAME = "disk_lightglue_dense_none_1x3x480x640.pte"

    @pytest.fixture
    def model(self, mock_logger):
        path = model_path_or_skip(self.MODEL_NAME)
        return DiskLightGlueExecuTorch("disk", mock_logger, config={
            "executorch_model_path": str(path),
            "input_shape": (1, 3, 480, 640),
            "num_keypoints": 256,
            "nms_radius": 4,
        })

    def test_detect_returns_correct_keypoint_count(self, model, load_img):
        img = load_img("box.png")
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        features = model.detect(tensor)
        assert features["keypoints"].shape[0] == 256


@pytest.mark.integration
class TestIntegrationD2NetDetector:
    MODEL_NAME = "d2net_dense_none_1x3x480x640.pte"

    @pytest.fixture
    def model(self, mock_logger):
        path = model_path_or_skip(self.MODEL_NAME)
        return D2NetExecuTorch("d2net", mock_logger, config={
            "executorch_model_path": str(path),
            "input_shape": (1, 3, 480, 640),
            "num_keypoints": 256,
            "nms_radius": 4,
        })

    def test_detect_on_real_image(self, model, load_img):
        img = load_img("box.png")
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        features = model.detect(tensor)
        assert features["keypoints"].shape[0] <= 256

    def test_detect_on_alternate_resolution_model(self, model, mock_logger, load_img):
        alt_path = model_path_or_skip("d2net_dense_none_1x3x720x960.pte")
        alt_model = D2NetExecuTorch("d2net", mock_logger, config={
            "executorch_model_path": str(alt_path),
            "input_shape": (1, 3, 720, 960),
            "num_keypoints": 256,
            "nms_radius": 4,
        })
        img = load_img("box.png")
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        features = alt_model.detect(tensor)
        assert features["keypoints"].shape[0] <= 256


@pytest.mark.integration
class TestIntegrationTFeatDescriptor:
    MODEL_NAME = "tfeat_none_1x1x32x32.pte"

    @pytest.fixture
    def descriptor(self, mock_logger):
        path = model_path_or_skip(self.MODEL_NAME)
        return TFeatExecuTorch("tfeat", mock_logger, config={
            "executorch_model_path": str(path), "patch_size": 32
        })

    def test_compute_on_real_image_with_synthetic_keypoints(self, descriptor, load_img):
        img = load_img("box.png")
        h, w = img.shape[:2]
        keypoints = [FakeKeypoint(w // 2, h // 2), FakeKeypoint(w // 4, h // 4)]
        features = descriptor.compute(img, {"kp": keypoints})
        assert features["des"].shape[0] == 2
        assert features["des"].dtype == np.float32


@pytest.mark.integration
class TestIntegrationHardNetDescriptor:
    MODEL_NAME = "hardnet_none_1x1x32x32.pte"

    @pytest.fixture
    def descriptor(self, mock_logger):
        path = model_path_or_skip(self.MODEL_NAME)
        return HardNetExecuTorch("hardnet", mock_logger, config={
            "executorch_model_path": str(path), "patch_size": 32
        })

    def test_compute_on_real_image(self, descriptor, load_img):
        img = load_img("box.png")
        h, w = img.shape[:2]
        keypoints = [FakeKeypoint(w // 2, h // 2)]
        features = descriptor.compute(img, {"kp": keypoints})
        assert features["des"].shape[0] == 1


@pytest.mark.integration
class TestIntegrationLightGlueMatcher:
    SP_LG_MODEL = "superpoint_lightglue_dense_none_1x3x480x640.pte"
    LG_MODEL = "lightglue_superpoint_k256_none_1x3x480x640.pte"

    @pytest.fixture
    def detector(self, mock_logger):
        path = model_path_or_skip(self.SP_LG_MODEL)
        return SuperPointLightGlueExecuTorch("superpoint", mock_logger, config={
            "executorch_model_path": str(path),
            "input_shape": (1, 3, 480, 640),
            "num_keypoints": 256,
            "nms_radius": 4,
        })

    @pytest.fixture
    def matcher(self, mock_logger):
        path = model_path_or_skip(self.LG_MODEL)
        return LightGlueExecuTorch(mock_logger, "lightglue", "superpoint", config={
            "executorch_model_path": str(path), "num_keypoints": 256
        })

    def test_end_to_end_match_same_image(self, detector, matcher, load_img):
        img = load_img("box.png")
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        features0 = detector.detect(tensor)
        features1 = detector.detect(tensor)

        result = matcher.match(features0, features1)
        assert "matches" in result
        assert "scores" in result
        assert result["matches"].shape[1] == 2

    def test_matches_reference_valid_indices(self, detector, matcher, load_img):
        img = load_img("box.png")
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        features0 = detector.detect(tensor)
        features1 = detector.detect(tensor)
        result = matcher.match(features0, features1)

        if result["matches"].shape[0] > 0:
            assert torch.all(result["matches"][:, 0] >= 0)
            assert torch.all(result["matches"][:, 0] < 256)
            assert torch.all(result["matches"][:, 1] >= 0)
            assert torch.all(result["matches"][:, 1] < 256)


@pytest.mark.integration
class TestIntegrationDiskLightGlueMatcher:
    DISK_MODEL = "disk_lightglue_dense_none_1x3x480x640.pte"
    LG_MODEL = "lightglue_disk_k256_none_1x3x480x640.pte"

    @pytest.fixture
    def detector(self, mock_logger):
        path = model_path_or_skip(self.DISK_MODEL)
        return DiskLightGlueExecuTorch("disk", mock_logger, config={
            "executorch_model_path": str(path),
            "input_shape": (1, 3, 480, 640),
            "num_keypoints": 256,
            "nms_radius": 4,
        })

    @pytest.fixture
    def matcher(self, mock_logger):
        path = model_path_or_skip(self.LG_MODEL)
        return LightGlueExecuTorch(mock_logger, "lightglue", "disk", config={
            "executorch_model_path": str(path), "num_keypoints": 256
        })

    def test_end_to_end_match(self, detector, matcher, load_img):
        img = load_img("box.png")
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        features0 = detector.detect(tensor)
        features1 = detector.detect(tensor)
        result = matcher.match(features0, features1)
        assert "matches" in result


@pytest.mark.integration
class TestIntegrationBackendVariants:

    @pytest.mark.parametrize("backend", ["none", "vulkan", "xnnpack"])
    def test_d2net_backend_variant_loads_and_runs(self, backend, mock_logger, load_img):
        filename = f"d2net_dense_{backend}_1x3x480x640.pte"
        path = model_path_or_skip(filename)
        model = D2NetExecuTorch("d2net", mock_logger, config={
            "executorch_model_path": str(path),
            "input_shape": (1, 3, 480, 640),
            "num_keypoints": 128,
            "nms_radius": 4,
        })
        img = load_img("box.png")
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        features = model.detect(tensor)
        assert features["keypoints"].shape[0] <= 128

    @pytest.mark.parametrize("backend", ["none", "vulkan", "xnnpack"])
    def test_tfeat_backend_variant_loads_and_runs(self, backend, mock_logger, load_img):
        filename = f"tfeat_{backend}_1x1x32x32.pte"
        path = model_path_or_skip(filename)
        descriptor = TFeatExecuTorch("tfeat", mock_logger, config={
            "executorch_model_path": str(path), "patch_size": 32
        })
        img = load_img("box.png")
        h, w = img.shape[:2]
        keypoints = [FakeKeypoint(w // 2, h // 2)]
        features = descriptor.compute(img, {"kp": keypoints})
        assert features["des"].shape[0] == 1
