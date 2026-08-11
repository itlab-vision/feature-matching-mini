import pytest
import cv2 as cv
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
from logging import Logger

from src.descriptors import Descriptor
from src.loftr import LoFTR
from src.detectors import Detector
from src.matchers import Matcher


@pytest.fixture
def mock_logger():
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

@pytest.fixture(scope="session")
def session_logger():
    return MagicMock(spec=Logger)


@pytest.fixture(scope="session")
def loftr_instance(session_logger):
    return LoFTR("loftr", logger=session_logger, config={'device': 'cpu'})


class TestLoFTRRegistration:
    def test_registered_in_factories(self):
        assert "loftr" in Detector._METHODS
        assert "loftr" in Descriptor._METHODS
        assert "loftr" in Matcher._METHODS

    def test_factory_creation(self, mock_logger):
        obj = Matcher.create("loftr", mock_logger, descriptor_name="loftr", config={'device': 'cpu'})
        assert isinstance(obj, LoFTR)
        assert obj.default_norm == cv.NORM_L2

    def test_factory_creation_with_config(self, mock_logger):
        obj = Detector.create("loftr", mock_logger, config={'threshold': 0.01, 'weights': 'outdoor'})
        assert isinstance(obj, LoFTR)
        assert obj._threshold == 0.01
        assert obj._weights_type == 'outdoor'


class TestLoFTRConfig:
    def test_default_threshold(self, mock_logger):
        loftr = LoFTR("loftr", mock_logger, config=None)
        assert loftr._threshold == 0.005

    def test_custom_threshold(self, mock_logger):
        loftr = LoFTR("loftr", mock_logger, config={'threshold': 0.05})
        assert loftr._threshold == 0.05

    def test_custom_device(self, mock_logger):
        loftr = LoFTR("loftr", mock_logger, config={'device': 'cpu'})
        assert loftr._device.type == 'cpu'

    def test_empty_config(self, mock_logger):
        loftr = LoFTR("loftr", mock_logger, config={})
        assert loftr._threshold == 0.005

    def test_custom_config(self, mock_logger):
        loftr = LoFTR(
            "loftr",
            mock_logger,
            config={
                'device': 'cpu',
                'weights': 'outdoor',
                'threshold': 0.01
            }
        )
        assert loftr._threshold == 0.01
        assert loftr._weights_type == 'outdoor'
        assert loftr._device.type == 'cpu'


class TestLoFTRSingleton:
    def test_shared_model_weights(self, session_logger):
        m1 = LoFTR("loftr", session_logger)
        m2 = LoFTR("loftr", session_logger)
        assert m1._model is m2._model
        assert id(m1._model) == id(m2._model)

    def test_eval_mode_persistence(self, loftr_instance):
        assert not loftr_instance._model.training

    def test_device_consistency(self, loftr_instance):
        model_device = next(loftr_instance._model.parameters()).device
        assert loftr_instance._device.type == model_device.type

    def test_singleton_after_deletion(self, session_logger, load_img):
        m1 = LoFTR("loftr", session_logger)
        m2 = LoFTR("loftr", session_logger)
        del m1

        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        try:
            m2.detect(img)
        except Exception as e:
            pytest.fail(f"Singleton model was corrupted after object deletion: {e}")



class TestLoFTRInference:
    def test_detect_returns_required_keys(self, loftr_instance, load_img):
        img = load_img("box.png")
        result = loftr_instance.detect(img)
        assert isinstance(result, dict)
        assert 'image' in result

    def test_match_output_structure(self, loftr_instance, load_img):
        img = load_img("box.png")
        feat = loftr_instance.detect(img)
        res = loftr_instance.match(feat, feat)

        for key in ['keypoints0', 'keypoints1', 'matches', 'scores']:
            assert key in res
            assert isinstance(res[key], torch.Tensor)

        assert res['matches'].dtype == torch.long
        assert res['scores'].dtype == torch.float32

    def test_coordinate_mapping_accuracy(self, loftr_instance, load_img):
        img = load_img("box.png")
        h, w = img.shape[:2]
        feat = loftr_instance.detect(img)
        res = loftr_instance.match(feat, feat)

        kp0 = res['keypoints0']

        assert torch.max(kp0[:, 0]) > 1.0 or len(kp0) == 0
        assert torch.all(kp0[:, 0] >= 0) and torch.all(kp0[:, 0] < w)
        assert torch.all(kp0[:, 1] >= 0) and torch.all(kp0[:, 1] < h)

    def test_num_features_parameter(self, session_logger, load_img):
        model = LoFTR("loftr", session_logger, config={'threshold': 0.9999, 'device': 'cpu'})
        img = load_img("box.png")
        res = model.match(model.detect(img), model.detect(img))
        assert len(res['matches']) <= 256

    def test_scores_confidence_range(self, loftr_instance, load_img):
        img = load_img("box.png")
        res = loftr_instance.match(loftr_instance.detect(img), loftr_instance.detect(img))
        if len(res['scores']) > 0:
            assert torch.all(res['scores'] >= 0.0)
            assert torch.all(res['scores'] <= 1.0)

    def test_match_logic_self_similarity(self, loftr_instance, load_img):
        img = load_img("box.png")
        feat = loftr_instance.detect(img)
        res = loftr_instance.match(feat, feat)

        matches = res['matches']
        for i in range(min(5, len(matches))):
            assert matches[i][0] == matches[i][1]


class TestLoFTRRobustness:
    def test_invalid_input_empty_dict(self, loftr_instance):
        res = loftr_instance.match({}, {})
        assert len(res['matches']) == 0

    def test_none_image_handling(self, loftr_instance, session_logger):
        res = loftr_instance.match({'image': None}, {'image': None})
        assert len(res['matches']) == 0
        assert session_logger.error.called

    def test_different_resolutions(self, loftr_instance, load_img):
        img1 = load_img("box.png")
        img2 = cv.resize(img1, (200, 150))

        f1 = loftr_instance.detect(img1)
        f2 = loftr_instance.detect(img2)

        try:
            res = loftr_instance.match(f1, f2)
            assert isinstance(res, dict)
        except Exception as e:
            pytest.fail(f"LoFTR failed on different resolutions: {e}")

    def test_grayscale_input_conversion(self, loftr_instance, load_img):
        img_gray = load_img("box.png", color=False)
        assert len(img_gray.shape) == 2
        f = loftr_instance.detect(img_gray)
        res = loftr_instance.match(f, f)
        assert len(res['matches']) > 0

    def test_very_small_images(self, loftr_instance):
        img = np.random.randint(0, 255, (14, 14, 3), dtype=np.uint8)
        f = loftr_instance.detect(img)
        res = loftr_instance.match(f, f)
        assert 'matches' in res

    def test_non_square_images(self, loftr_instance):
        img = np.random.randint(0, 255, (100, 600, 3), dtype=np.uint8)
        f = loftr_instance.detect(img)
        res = loftr_instance.match(f, f)
        assert len(res['keypoints0']) > 0

    def test_reproducibility_on_cpu(self, loftr_instance, load_img):
        img = load_img("box.png")
        f = loftr_instance.detect(img)
        res1 = loftr_instance.match(f, f)
        res2 = loftr_instance.match(f, f)

        assert torch.equal(res1['matches'], res2['matches'])
        assert torch.allclose(res1['scores'], res2['scores'], atol=1e-6)
