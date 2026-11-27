import logging
from pathlib import Path
from unittest.mock import MagicMock

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


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / 'models'


def build_config(backend, extractor_model, matcher_model):
    extractor_config = {
        'backend': backend,
        'nfeatures': 512,
        'threshold': -1e9,
    }
    matcher_config = {
        'backend': backend,
        'normalize_keypoints': True,
    }
    if backend == 'executorch':
        extractor_config.update({
            'executorch_model_path': str(extractor_model),
            'input_shape': [1, 3, 480, 640],
            'num_keypoints': 512,
        })
        matcher_config.update({
            'executorch_model_path': str(matcher_model),
            'num_keypoints': 512,
        })
    else:
        extractor_config['superpoint_model_path'] = str(extractor_model)
        matcher_config['lightglue_model_path'] = str(matcher_model)

    return {
        'stitcher_type': 'DetailedStitcher',
        'detector': 'superpoint_lightglue',
        'descriptor': 'superpoint_lightglue',
        'matcher': 'lightglue',
        'sequential_matching': True,
        'src_config': {
            'detector': extractor_config,
            'descriptor': extractor_config.copy(),
            'matcher': matcher_config,
        },
        'try_cuda': False,
        'work_megapix': 0.6,
        'seam_megapix': 0.1,
        'compose_megapix': -1,
        'range_width': -1,
        'estimator': 'homography',
        'match_conf': 0.3,
        'conf_thresh': 0.3,
        'ba': 'ray',
        'ba_refine_mask': 'xxxxx',
        'wave_correct': 'horiz',
        'warp': 'spherical',
        'seam': 'gc_color',
        'expos_comp': 'gain_blocks',
        'expos_comp_nr_feeds': 1,
        'expos_comp_nr_filtering': 2,
        'expos_comp_block_size': 32,
        'blend': 'multiband',
        'blend_strength': 5,
    }


def close_stitcher_models(stitcher):
    if hasattr(stitcher.features, 'detector'):
        stitcher.features.detector.close()
    if hasattr(stitcher.features, 'descriptor'):
        stitcher.features.descriptor.close()
    if stitcher.matcher._matcher is not None:
        stitcher.matcher._matcher.close()


@pytest.mark.parametrize('backend, extractor_name, matcher_name', [
    ('onnx', 'superpoint_lightglue.onnx', 'lightglue_superpoint_k512.onnx'),
    ('opencv', 'superpoint_lightglue.onnx', 'lightglue_superpoint_opencv.onnx'),
    ('executorch', 'superpoint_lightglue_xnnpack.pte', 'lightglue_superpoint_k512_xnnpack.pte'),
])
def test_superpoint_lightglue_panorama_smoke(backend, extractor_name, matcher_name):
    extractor_model = MODELS_DIR / extractor_name
    matcher_model = MODELS_DIR / matcher_name
    missing_models = [str(path) for path in (extractor_model, matcher_model) if not path.is_file()]
    if missing_models:
        pytest.skip(f'Model files are not available: {missing_models}')
    if backend == 'executorch':
        pytest.importorskip('executorch.runtime')

    images = [cv2.imread(str(ROOT_DIR / 'test_data' / name))
              for name in ('box.png', 'box_in_scene.png')]
    assert all(image is not None for image in images)

    stitcher = PanoramaStitcher.create(
        MagicMock(spec=logging.Logger),
        'DetailedStitcher',
        build_config(backend, extractor_model, matcher_model),
    )
    try:
        panorama = stitcher.stitch(images)
    finally:
        close_stitcher_models(stitcher)

    assert isinstance(panorama, np.ndarray)
    assert panorama.ndim == 3 and panorama.shape[2] == 3
    assert panorama.size > 0
    assert panorama.shape[1] >= max(image.shape[1] for image in images)
    assert np.any(panorama)
    assert stitcher.perf_report['num_images'] == 2
    assert stitcher.perf_report['panorama_resolution'] == {
        'width': panorama.shape[1],
        'height': panorama.shape[0],
    }
    assert stitcher.perf_report['stage_times_sec']['extract features'] > 0
    assert stitcher.perf_report['stage_times_sec']['match features'] > 0
