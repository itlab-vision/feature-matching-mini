import pytest
import cv2 as cv
import torch
from pathlib import Path
from unittest.mock import MagicMock
from logging import Logger

from src.lightglue_pipeline import LightGlueFeatureExtractor
from src.lightglue_matcher import LightGlueMatcher
from src.detectors import Detector
from src.descriptors import Descriptor
from src.matchers import Matcher


@pytest.fixture
def mock_logger():
    return MagicMock(spec=Logger)


@pytest.fixture(scope="session")
def session_logger():
    return MagicMock(spec=Logger)


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


@pytest.fixture
def to_tensor():
    def _convert(img):
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return tensor
    return _convert


@pytest.fixture(scope="session")
def lg_instance(session_logger):
    return LightGlueFeatureExtractor("superpoint_lightglue",
                                     logger=session_logger, config=None)


class TestLightGlueFeatureExtractorRegistration:
    def test_all_extractors_registered_in_detector(self):
        expected = {"superpoint_lightglue", "disk_lightglue", "sift_lightglue",
                    "aliked_lightglue", "doghardnet_lightglue"}
        assert expected.issubset(Detector._METHODS.keys())

    def test_all_extractors_registered_in_descriptor(self):
        expected = {"superpoint_lightglue", "disk_lightglue", "sift_lightglue",
                    "aliked_lightglue", "doghardnet_lightglue"}
        assert expected.issubset(Descriptor._METHODS.keys())

    def test_factory_creation(self, mock_logger):
        obj = Detector.create("superpoint_lightglue", mock_logger)
        assert isinstance(obj, LightGlueFeatureExtractor)
        assert obj.default_norm == cv.NORM_L2

    def test_factory_creation_with_config(self, mock_logger):
        obj = Detector.create("superpoint_lightglue", mock_logger,
                              config={'device': 'cpu', 'max_num_keypoints': 512})
        assert isinstance(obj, LightGlueFeatureExtractor)
        assert obj._device.type == 'cpu'


class TestLightGlueFeatureExtractorConfig:
    def test_default_device(self, mock_logger):
        lg = LightGlueFeatureExtractor("superpoint_lightglue", mock_logger, config=None)
        assert lg._device is not None

    def test_explicit_cpu_device(self, mock_logger):
        lg = LightGlueFeatureExtractor("superpoint_lightglue", mock_logger,
                                       config={'device': 'cpu'})
        assert lg._device.type == 'cpu'

    def test_empty_config(self, mock_logger):
        lg = LightGlueFeatureExtractor("superpoint_lightglue", mock_logger, config={})
        assert isinstance(lg, LightGlueFeatureExtractor)

    def test_none_config(self, mock_logger):
        lg = LightGlueFeatureExtractor("superpoint_lightglue", mock_logger, config=None)
        assert isinstance(lg, LightGlueFeatureExtractor)

    def test_unknown_extractor_raises(self, mock_logger):
        with pytest.raises(ValueError, match="not found"):
            LightGlueFeatureExtractor("unknown_lightglue", mock_logger, config=None)


class TestLightGlueFeatureExtractorSingleton:
    def test_shared_model_same_extractor(self, mock_logger):
        lg1 = LightGlueFeatureExtractor("superpoint_lightglue", mock_logger, config=None)
        lg2 = LightGlueFeatureExtractor("superpoint_lightglue", mock_logger, config=None)
        assert lg1._extractor is lg2._extractor

    def test_different_extractors_different_models(self, mock_logger):
        lg_sp = LightGlueFeatureExtractor("superpoint_lightglue", mock_logger, config=None)
        lg_sift = LightGlueFeatureExtractor("sift_lightglue", mock_logger, config=None)
        assert lg_sp._extractor is not lg_sift._extractor

    def test_eval_mode(self, mock_logger):
        lg = LightGlueFeatureExtractor("superpoint_lightglue", mock_logger, config=None)
        assert not lg._extractor.training

    def test_device_consistency(self, mock_logger):
        lg = LightGlueFeatureExtractor("superpoint_lightglue", mock_logger,
                                       config={'device': 'cpu'})
        model_device = next(lg._extractor.parameters()).device
        assert lg._device.type == model_device.type


class TestLightGlueFeatureExtractorInference:
    def test_detect_returns_dict(self, lg_instance, load_img, to_tensor):
        img = load_img("box.png")
        tensor = to_tensor(img)
        result = lg_instance.detect(tensor)
        assert isinstance(result, dict)
        assert 'keypoints' in result

    def test_forward_none_input(self, lg_instance, session_logger):
        result = lg_instance._forward(None)
        assert result == {'descriptors': (), 'keypoints': ()}
        session_logger.error.assert_called()

    def test_detect_sets_is_extracted(self, lg_instance, load_img, to_tensor):
        img = load_img("box.png")
        tensor = to_tensor(img)
        lg_instance.detect(tensor)
        assert LightGlueFeatureExtractor._is_extracted is True

    def test_compute_uses_cache_after_detect(self, lg_instance, load_img, to_tensor):
        img = load_img("box.png")
        tensor = to_tensor(img)

        detected = lg_instance.detect(tensor)
        computed = lg_instance.compute(tensor)

        assert detected is computed
        assert LightGlueFeatureExtractor._is_extracted is False

    def test_compute_without_detect_runs_forward(self, lg_instance, load_img, to_tensor):
        img = load_img("box.png")
        tensor = to_tensor(img)

        LightGlueFeatureExtractor._is_extracted = False
        result = lg_instance.compute(tensor)
        assert isinstance(result, dict)

    def test_detect_and_compute_returns_dict(self, lg_instance, load_img, to_tensor):
        img = load_img("box.png")
        tensor = to_tensor(img)
        result = lg_instance.detectAndCompute(tensor)
        assert isinstance(result, dict)
        assert 'keypoints' in result

    def test_3d_tensor_gets_batch_dim(self, lg_instance, load_img, to_tensor):
        img = load_img("box.png")
        tensor = to_tensor(img)
        assert tensor.ndim == 3
        result = lg_instance._forward(tensor)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("extractor_name", [
        "superpoint_lightglue", "sift_lightglue"
    ])
    def test_multiple_extractors_inference(self, extractor_name, mock_logger, load_img, to_tensor):
        img = load_img("box.png")
        tensor = to_tensor(img)
        lg = LightGlueFeatureExtractor(extractor_name, mock_logger, config={'device': 'cpu'})
        result = lg.detectAndCompute(tensor)
        assert isinstance(result, dict)
        assert 'keypoints' in result


class TestLightGlueMatcherRegistration:
    def test_registered_in_matcher(self):
        assert "lightglue" in Matcher._METHODS

    def test_factory_creation(self, session_logger):
        obj = Matcher.create("lightglue", session_logger,
                             descriptor_name="superpoint_lightglue",
                             config={'device': 'cpu'})
        assert isinstance(obj, LightGlueMatcher)
        assert obj._extractor_name == 'superpoint'


class TestLightGlueMatcherConfig:
    def test_extractor_type_from_descriptor_name(self, session_logger):
        matcher = Matcher.create("lightglue", session_logger,
                                 descriptor_name="sift_lightglue",
                                 config={'device': 'cpu'})
        assert matcher._extractor_name == 'sift'

    def test_device_cpu(self, session_logger):
        matcher = Matcher.create("lightglue", session_logger,
                                 descriptor_name="superpoint_lightglue",
                                 config={'device': 'cpu'})
        assert matcher._device.type == 'cpu'

    def test_eval_mode(self, session_logger):
        matcher = Matcher.create("lightglue", session_logger,
                                 descriptor_name="superpoint_lightglue",
                                 config={'device': 'cpu'})
        assert not matcher._matcher.training

    def test_none_config(self, session_logger):
        matcher = LightGlueMatcher(session_logger, "lightglue",
                                   "superpoint_lightglue", config=None)
        assert isinstance(matcher, LightGlueMatcher)

    def test_empty_config(self, session_logger):
        matcher = LightGlueMatcher(session_logger, "lightglue",
                                   "superpoint_lightglue", config={})
        assert isinstance(matcher, LightGlueMatcher)


class TestLightGlueMatcherInference:
    def test_match_returns_tensor(self, session_logger, lg_instance, load_img, to_tensor):
        img1 = load_img("box.png")
        img2 = load_img("box_in_scene.png")

        matcher = Matcher.create("lightglue", session_logger,
                                 descriptor_name="superpoint_lightglue",
                                 config=None)

        f0 = lg_instance.detectAndCompute(to_tensor(img1))
        f1 = lg_instance.detectAndCompute(to_tensor(img2))

        result = matcher.match(f0, f1)
        assert isinstance(result, dict)
        assert 'matches' in result
        assert 'scores' in result
        assert isinstance(result['matches'], torch.Tensor)
        assert isinstance(result['scores'], torch.Tensor)

    def test_match_shape(self, session_logger, lg_instance, load_img, to_tensor):
        img1 = load_img("box.png")
        img2 = load_img("box_in_scene.png")

        matcher = Matcher.create("lightglue", session_logger,
                                 descriptor_name="superpoint_lightglue",
                                 config=None)

        f0 = lg_instance.detectAndCompute(to_tensor(img1))
        f1 = lg_instance.detectAndCompute(to_tensor(img2))

        result = matcher.match(f0, f1)
        assert result['matches'].ndim == 2
        assert result['matches'].shape[1] == 2

    def test_match_reproducibility(self, session_logger, lg_instance, load_img, to_tensor):
        img1 = load_img("box.png")
        img2 = load_img("box_in_scene.png")

        matcher = Matcher.create("lightglue", session_logger,
                                 descriptor_name="superpoint_lightglue",
                                 config=None)

        f0 = lg_instance.detectAndCompute(to_tensor(img1))
        f1 = lg_instance.detectAndCompute(to_tensor(img2))

        result1 = matcher.match(f0, f1)
        result2 = matcher.match(f0, f1)

        assert torch.equal(result1['matches'], result2['matches'])
        assert torch.equal(result1['scores'], result2['scores'])
