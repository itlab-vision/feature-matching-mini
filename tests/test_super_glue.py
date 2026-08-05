import pytest
import cv2 as cv
import torch
from pathlib import Path
from unittest.mock import MagicMock
from logging import Logger

from src.super_point import SuperPoint
from src.super_glue import SuperGlueMatcher
from src.matchers import Matcher


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
def sp_instance(session_logger):
    return SuperPoint("superpoint", logger=session_logger, config=None)


class TestSuperGlueRegistration:
    def test_registered_in_matcher(self):
        assert "superglue" in Matcher._METHODS

    def test_factory_creation(self, session_logger):
        obj = Matcher.create("superglue", session_logger,
                             descriptor_name="superpoint",
                             config=None)
        assert isinstance(obj, SuperGlueMatcher)
        assert obj.descriptor_name == 'superpoint'


class TestSuperGlueMatcherConfig:
    def test_extractor_type_from_descriptor_name(self, session_logger):
        matcher = Matcher.create("superglue", session_logger,
                                 descriptor_name="superpoint",
                                 config={'device': 'cpu'})
        assert matcher.descriptor_name == 'superpoint'

    def test_device_cpu(self, session_logger):
        matcher = Matcher.create("superglue", session_logger,
                                 descriptor_name="superpoint",
                                 config={'device': 'cpu'})
        assert matcher._device.type == 'cpu'

    def test_eval_mode(self, session_logger):
        matcher = Matcher.create("superglue", session_logger,
                                 descriptor_name="superpoint",
                                 config=None)
        assert not matcher._matcher.training

    def test_none_config(self, session_logger):
        matcher = SuperGlueMatcher(session_logger, "superglue",
                                   "superpoint", config=None)
        assert isinstance(matcher, SuperGlueMatcher)

    def test_empty_config(self, session_logger):
        matcher = SuperGlueMatcher(session_logger, "superglue",
                                   "superpoint", config={})
        assert isinstance(matcher, SuperGlueMatcher)

    def test_weights_switching(self, session_logger):
        for w_type in ['indoor', 'outdoor']:
            matcher = SuperGlueMatcher(session_logger, "superglue",
                                       "superpoint", config={'weights': w_type})
            assert matcher._matcher.config['weights'] == w_type


class TestSuperGlueMatcherInference:
    def test_match_returns_tensor(self, session_logger, sp_instance, load_img, to_tensor):
        img1 = load_img("box.png")
        img2 = load_img("box_in_scene.png")

        matcher = Matcher.create("superglue", session_logger,
                                 descriptor_name="superpoint",
                                 config=None)

        f0 = sp_instance.detectAndCompute(to_tensor(img1))
        f1 = sp_instance.detectAndCompute(to_tensor(img2))

        result = matcher.match(f0, f1)
        assert isinstance(result, dict)
        assert 'matches' in result
        assert 'scores' in result
        assert isinstance(result['matches'], torch.Tensor)
        assert isinstance(result['scores'], torch.Tensor)

    def test_match_shape(self, session_logger, sp_instance, load_img, to_tensor):
        img1 = load_img("box.png")
        img2 = load_img("box_in_scene.png")

        matcher = Matcher.create("superglue", session_logger,
                                 descriptor_name="superpoint",
                                 config=None)

        f0 = sp_instance.detectAndCompute(to_tensor(img1))
        f1 = sp_instance.detectAndCompute(to_tensor(img2))

        result = matcher.match(f0, f1)
        assert result['matches'].ndim == 2
        assert result['matches'].shape[1] == 2

    def test_match_reproducibility(self, session_logger, sp_instance, load_img, to_tensor):
        img1 = load_img("box.png")
        img2 = load_img("box_in_scene.png")

        matcher = Matcher.create("superglue", session_logger,
                                 descriptor_name="superpoint",
                                 config=None)

        f0 = sp_instance.detectAndCompute(to_tensor(img1))
        f1 = sp_instance.detectAndCompute(to_tensor(img2))

        result1 = matcher.match(f0, f1)
        result2 = matcher.match(f0, f1)

        assert torch.equal(result1['matches'], result2['matches'])
        assert torch.equal(result1['scores'], result2['scores'])

    def test_match_zero_keypoints(self, session_logger):
        matcher = SuperGlueMatcher(session_logger, "superglue",
                                   "superpoint", config={'device': 'cpu'})

        f0 = {'keypoints': torch.empty((0, 2)), 'descriptors': torch.empty((256, 0)),
              'scores': torch.empty(0), 'width': 640, 'height': 480}
        f1 = {'keypoints': torch.empty((0, 2)), 'descriptors': torch.empty((256, 0)),
              'scores': torch.empty(0), 'width': 640, 'height': 480}

        result = matcher.match(f0, f1)
        assert len(result['matches']) == 0
        assert len(result['scores']) == 0

    def test_threshold_impact(self, session_logger, sp_instance, load_img, to_tensor):
        img1 = load_img("box.png")
        img2 = load_img("box_in_scene.png")
        f0 = sp_instance.detectAndCompute(to_tensor(img1))
        f1 = sp_instance.detectAndCompute(to_tensor(img2))

        strict_matcher = SuperGlueMatcher(session_logger, "superglue", "superpoint",
                                          config={'threshold': 0.99})
        loose_matcher = SuperGlueMatcher(session_logger, "superglue", "superpoint",
                                         config={'threshold': 0.01})

        res_strict = strict_matcher.match(f0, f1)
        res_loose = loose_matcher.match(f0, f1)
        assert len(res_loose['matches']) >= len(res_strict['matches'])
