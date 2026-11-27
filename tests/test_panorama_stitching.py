import logging
from pathlib import Path
from unittest.mock import MagicMock

import cv2 as cv
import numpy as np
import pytest

from stitching.panorama_stitcher import PanoramaStitcher


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

    images = [cv.imread(str(ROOT_DIR / 'test_data' / name))
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
