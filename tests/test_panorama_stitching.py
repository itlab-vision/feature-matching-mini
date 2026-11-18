import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

from stitching.panorama_stitcher import PanoramaStitcher
from stitching.panorama_utils.feature_extractor import FeatureExtractor, PipelineFeatureExtractor
from stitching.read_write_data import YamlReader


def test_src_sift_stitches_overlapping_images_and_drops_disconnected_frame():
    logger = logging.getLogger(__name__)
    config_path = (Path(__file__).resolve().parents[1] / 'samples' / 'configs'
                   / 'detailed_stitching_src_opencv_config.yaml')
    config = YamlReader(logger, config_path).get_config()[0]
    rng = np.random.default_rng(155)
    texture = rng.integers(0, 255, (160, 360, 3), dtype=np.uint8)
    canvas = cv2.resize(texture, (1080, 480))
    left = canvas[:, :640].copy()
    right = canvas[:, 220:860].copy()
    unrelated = np.full_like(left, 127)
    stitcher = PanoramaStitcher.create(logger, config['stitcher_type'], config)

    baseline = stitcher.stitch([left, right])
    panorama = stitcher.stitch([unrelated, left, right])

    assert baseline.shape[1] > left.shape[1]
    assert panorama.shape == pytest.approx(baseline.shape, abs=4)
    assert stitcher.perf_report['num_images'] == 2

    with pytest.raises(ValueError, match='at least two overlapping images'):
        stitcher.stitch([unrelated, unrelated])


@pytest.mark.parametrize(('name', 'dtype'), [
    ('orb', np.uint8),
    ('sift', np.float32),
    ('brisk', np.uint8),
    ('akaze', np.uint8),
])
def test_legacy_feature_name_produces_opencv_image_features(name, dtype):
    texture = np.random.default_rng(42).integers(0, 256, (80, 100, 3), dtype=np.uint8)
    image = cv2.resize(texture, (400, 320))
    extractor = FeatureExtractor.create(name, {'features': name}, logging.getLogger(__name__))

    features = extractor.extract(image)

    assert isinstance(extractor, PipelineFeatureExtractor)
    assert isinstance(features, cv2.detail.ImageFeatures)
    assert features.img_size == (400, 320)
    assert features.keypoints
    descriptors = features.descriptors.get()
    assert descriptors.shape[0] == len(features.keypoints)
    assert descriptors.dtype == dtype


@pytest.mark.parametrize('mode', ['2nearest', '2nearest_range', 'affine'])
def test_two_nearest_modes_filter_matches_and_estimate_pair_geometry(mode):
    from stitching.panorama_utils.matchers import MatcherWrapper
    from stitching.panorama_utils.utils import src_features_to_image_features

    rng = np.random.default_rng(10)
    points = rng.uniform(20, 70, size=(12, 2)).astype(np.float32)
    descriptors = rng.random((12, 8), dtype=np.float32)
    # Duplicate descriptors give equally good neighbors and must fail the ratio test.
    descriptors[-1] = descriptors[-2]
    features = [
        src_features_to_image_features(
            {'keypoints': points + [7 * i, 3 * i], 'descriptors': descriptors},
            np.zeros((100, 110 + 20 * i), dtype=np.uint8),
        )
        for i in range(3)
    ]
    matcher = MatcherWrapper.create(mode, {'match_conf': 0.3, 'range_width': 1}, logging.getLogger(__name__))

    pairs = matcher.apply2(features)
    pair, reverse = pairs[1], pairs[3]
    assert len(pairs) == 9
    assert [(m.queryIdx, m.trainIdx) for m in pair.matches] == [(i, i) for i in range(10)]
    assert pair.num_inliers == reverse.num_inliers == 10
    assert np.all(pair.inliers_mask == 1)
    assert np.array_equal(reverse.inliers_mask, pair.inliers_mask)
    assert np.allclose(reverse.H @ pair.H, np.eye(3), atol=1e-6)
    expected = np.eye(3)
    expected[:2, 2] = (7, 3) if mode == 'affine' else (-3, 3)
    assert np.allclose(pair.H, expected, atol=1e-5)
    assert (pairs[2].H is None) == (mode == '2nearest_range')
