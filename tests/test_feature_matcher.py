import pytest
import cv2 as cv
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
from logging import Logger

from src.matchers import Matcher
from src.feature_matcher import FeatureMatcherCV2
from src.algorithms import DNN_ALGORITHMS, DESCRIPTOR_MATCHER_COMPATIBILITY


@pytest.fixture
def mock_logger():
    return MagicMock(spec=Logger)


@pytest.fixture
def load_img():
    def _load(name):
        path = Path(__file__).parent.parent / "test_data" / name
        img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
        if img is None:
            pytest.fail(f"Failed to load image. On path: {path.absolute()}")
        return img
    return _load


class TestFeatureMatcherInit:
    def test_init_params(self, mock_logger):
        matcher = FeatureMatcherCV2(
            logger=mock_logger,
            detector='orb',
            descriptor='orb',
            matcher='bf',
            config={'matcher': {'mode': 'knn'}}
        )
        assert matcher._detector == 'orb'
        assert matcher._descriptor == 'orb'
        assert matcher._matcher == 'bf'
        assert matcher._matcher_config.get('mode') == 'knn'

    def test_init_empty_config(self, mock_logger):
        matcher = FeatureMatcherCV2(logger=mock_logger, config={})
        assert matcher._detector == 'sift'
        assert matcher._descriptor == 'sift'

    def test_init_none_config(self, mock_logger):
        matcher = FeatureMatcherCV2(logger=mock_logger, config=None)
        assert matcher._detector == 'sift'

    def test_init_full_config(self, mock_logger):
        config = {
            'detector': {'nfeatures': 500},
            'descriptor': {'nfeatures': 500},
            'matcher': {'mode': 'knn'}
        }
        matcher = FeatureMatcherCV2(logger=mock_logger, config=config)
        assert matcher._detector_config == {'nfeatures': 500}
        assert matcher._descriptor_config == {'nfeatures': 500}
        assert matcher._matcher_config.get('mode') == 'knn'


class TestFeatureMatcherMatch:
    @pytest.mark.parametrize("det, des, mat", [
        ('sift', 'sift', 'bf'),
        ('orb', 'orb', 'bf'),
        ('sift', 'brisk', 'bf')
    ])
    def test_match_full_pipeline_success(self, det, des, mat, mock_logger, load_img):
        img1 = load_img("box.png")
        img2 = load_img("box_in_scene.png")

        matcher_cv = FeatureMatcherCV2(
            logger=mock_logger,
            detector=det,
            descriptor=des,
            matcher=mat,
            config={'matcher': {'mode': 'simple'}}
        )

        features0, features1, correspondences = matcher_cv.match(img1, img2)

        assert isinstance(features0, dict)
        assert isinstance(features1, dict)
        assert isinstance(correspondences, dict)
        assert 'matches' in correspondences
        assert len(correspondences.get('matches')) > 0

    def test_match_reproducibility(self, mock_logger, load_img):
        img1 = load_img("box.png")
        img2 = load_img("box_in_scene.png")
        matcher_cv = FeatureMatcherCV2(logger=mock_logger)

        _, _, corr1 = matcher_cv.match(img1, img2)
        _, _, corr2 = matcher_cv.match(img1, img2)
        assert len(corr1.get('matches')) == len(corr2.get('matches'))

    def test_match_with_detector_config(self, mock_logger, load_img):
        img1 = load_img("box.png")
        img2 = load_img("box_in_scene.png")

        limit = 100
        matcher_cv = FeatureMatcherCV2(
            logger=mock_logger,
            config={'detector': {'nfeatures': limit}}
        )
        features0, features1, correspondences = matcher_cv.match(img1, img2)
        kp = features0.get('kp')
        assert kp is not None
        assert len(kp) <= limit + 5

    def test_match_with_full_config(self, mock_logger, load_img):
        img1 = load_img("box.png")
        img2 = load_img("box_in_scene.png")

        config = {
            'detector': {'nfeatures': 200},
            'descriptor': {'nfeatures': 200},
            'matcher': {'mode': 'simple'}
        }
        matcher_cv = FeatureMatcherCV2(logger=mock_logger, config=config)
        features0, features1, correspondences = matcher_cv.match(img1, img2)
        assert isinstance(correspondences.get('matches'), (list, tuple))


class TestFeatureMatcherCompatibility:
    valid_combinations = [
        (det, des, mat, mode)
        for det, descriptors in FeatureMatcherCV2._DETECTOR_DESCRIPTOR_COMPATIBILITY.items()
        for des in descriptors
        for mat in Matcher._METHODS.keys()
        for mode in ['simple', 'knn']
        if det not in DNN_ALGORITHMS
        if mat in DESCRIPTOR_MATCHER_COMPATIBILITY.get(des, [])
    ]

    @pytest.mark.parametrize("det, des, mat, mode", valid_combinations)
    def test_known_compatible_pairs(self, det, des, mat, mode, mock_logger, load_img):
        if det in EXECUTORCH_ALGORITHMS or des in EXECUTORCH_ALGORITHMS or mat in EXECUTORCH_ALGORITHMS:
            pytest.skip("Model files in the torchexecut format are missing,"
                        " so we are skipping these combinations")

        img1 = load_img("box.png")
        img2 = load_img("box_in_scene.png")

        matcher_cv = FeatureMatcherCV2(
            logger=mock_logger,
            detector=det,
            descriptor=des,
            matcher=mat,
            config={'matcher': {'mode': mode}}
        )

        try:
            features0, features1, correspondences = matcher_cv.match(img1, img2)
            assert features0 is not None
            assert features1 is not None
            assert isinstance(correspondences, dict)
            assert 'matches' in correspondences

            matches = correspondences.get('matches')
            if mode == 'knn' and len(matches) > 0:
                assert isinstance(matches[0], (list, tuple))

        except Exception as e:
            pytest.fail(
                f"FAILED COMBINATION: \n"
                f"Detector: {det} | Descriptor: {des} | Matcher: {mat} | Mode: {mode}\n"
                f"Error: {e}"
            )

    def test_incompatible_descriptor_matcher_raises(self, mock_logger):
        with pytest.raises(ValueError, match="cannot be used with Matcher"):
            FeatureMatcherCV2(
                logger=mock_logger,
                detector='sift',
                descriptor='sift',
                matcher='lightglue'
            )

    def test_msd_sift_incompatibility(self, mock_logger):
        with pytest.raises(ValueError):
            FeatureMatcherCV2(
                logger=mock_logger,
                detector='msd',
                descriptor='sift'
            )


class TestFeatureMatcherRobustness:
    def test_match_raises_value_error_on_empty_image(self, mock_logger):
        matcher_cv = FeatureMatcherCV2(logger=mock_logger)
        empty_img = np.zeros((10, 10), dtype=np.uint8)

        with pytest.raises(ValueError, match="Failed to detect key points"):
            matcher_cv.match(empty_img, empty_img)

    def test_match_with_invalid_descriptor_input(self, mock_logger):
        matcher_cv = FeatureMatcherCV2(logger=mock_logger)
        try:
            _, _, correspondences = matcher_cv.match(None, None)
            if isinstance(correspondences, dict):
                assert len(correspondences.get('matches', [])) == 0
            else:
                assert len(correspondences) == 0
        except Exception:
            pass

    def test_invalid_config_key_ignored_or_raises(self, mock_logger, load_img):
        img1 = load_img("box.png")
        img2 = load_img("box_in_scene.png")

        with pytest.raises((TypeError, Exception, cv.error)):
            matcher_cv = FeatureMatcherCV2(
                logger=mock_logger,
                config={'detector': {'invalid_param': 999}}
            )
            matcher_cv.match(img1, img2)


class TestFeatureMatcherVisualization:
    def test_visualize_matches_shape(self, mock_logger, load_img):
        img1 = load_img("box.png")
        img2 = load_img("box_in_scene.png")

        matcher_cv = FeatureMatcherCV2(
            logger=mock_logger,
            config={'matcher': {'mode': 'simple'}}
        )
        features0, features1, correspondences = matcher_cv.match(img1, img2)
        res_img = matcher_cv.visualize_matches(img1, features0, img2, features1, correspondences)

        assert res_img.shape[0] >= max(img1.shape[0], img2.shape[0])
        assert res_img.shape[1] == img1.shape[1] + img2.shape[1]

    def test_visualize_no_matches(self, mock_logger, load_img):
        img1 = load_img("box.png")
        img2 = load_img("box_in_scene.png")

        matcher_cv = FeatureMatcherCV2(logger=mock_logger)
        empty_features = {'kp': [], 'des': None}
        empty_correspondences = {'matches': [], 'scores': None}
        res_img = matcher_cv.visualize_matches(img1, empty_features,
                                               img2, empty_features,
                                               empty_correspondences)

        assert res_img.shape[1] == img1.shape[1] + img2.shape[1]
        assert mock_logger.warning.called

    def test_visualize_knn_mode(self, mock_logger, load_img):
        img1 = load_img("box.png")
        img2 = load_img("box_in_scene.png")

        matcher_cv = FeatureMatcherCV2(
            logger=mock_logger,
            config={'matcher': {'mode': 'knn'}}
        )
        features0, features1, correspondences = matcher_cv.match(img1, img2)
        res_img = matcher_cv.visualize_matches(img1, features0, img2, features1, correspondences)
        assert res_img is not None


class TestFeatureMatcherInvalidInputs:

    def test_raises_value_error_on_invalid_detector(self, mock_logger):
        with pytest.raises(ValueError, match="Detector 'invalid_det' is not registered in compatibility matrix"):
            FeatureMatcherCV2(logger=mock_logger, detector='invalid_det').match(
                np.zeros((100, 100), dtype=np.uint8),
                np.zeros((100, 100), dtype=np.uint8)
            )

    def test_raises_value_error_on_invalid_matcher(self, mock_logger):
        with pytest.raises(ValueError):
            FeatureMatcherCV2(logger=mock_logger, matcher='unknown_matcher')
