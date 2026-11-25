import logging
from pathlib import Path
from unittest.mock import MagicMock

import cv2 as cv
import pytest

from src.algorithms import ALL_DESCRIPTORS, ALL_DETECTORS, ALL_MATCHERS, get_default_backend
from stitching.panorama_utils.feature_extractor import FeatureExtractor
from stitching.panorama_utils.matchers import MatcherWrapper


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / 'models'

METHOD_CASES = [
    ('sift', 'sift', 'bf'),
    ('orb', 'orb', 'flann'),
    ('fast', 'brisk', 'bf'),
    ('akaze', 'akaze', 'flann'),
    ('brisk', 'brief', 'bf'),
    ('kaze', 'kaze', 'flann'),
    ('gftt', 'freak', 'bf'),
    ('mser', 'daisy', 'flann'),
    ('agast', 'latch', 'bf'),
    ('blob', 'beblid', 'flann'),
    ('star', 'teblid', 'bf'),
    ('harrislaplace', 'vgg', 'flann'),
    ('msd', 'boostdesc', 'bf'),
    ('sift', 'tfeat', 'flann'),
    ('orb', 'hardnet', 'bf'),
    ('superpoint', 'superpoint', 'superglue'),
    ('superpoint_lightglue', 'superpoint_lightglue', 'lightglue'),
    ('disk_lightglue', 'disk_lightglue', 'lightglue'),
    ('disk', 'disk', 'bf'),
    ('sift_lightglue', 'sift_lightglue', 'lightglue'),
    ('aliked_lightglue', 'aliked_lightglue', 'lightglue'),
    ('aliked', 'aliked', 'bf'),
    ('doghardnet_lightglue', 'doghardnet_lightglue', 'lightglue'),
    ('d2net', 'd2net', 'bf'),
    ('xfeat', 'xfeat', 'bf'),
    ('r2d2', 'r2d2', 'bf'),
    ('loftr', 'loftr', 'loftr'),
    ('roma', 'roma', 'roma'),
]


def component_config(name, component_index):
    config = {'backend': get_default_backend(name, component_index)}
    if config['backend'] == 'torch':
        config['device'] = 'cpu'
    if component_index < 2:
        config['nfeatures'] = 128
    if name in ('sift_lightglue', 'doghardnet_lightglue'):
        config['threshold'] = 0.0
    if name == 'disk':
        config['disk_model_path'] = str(MODELS_DIR / 'disk_1024.onnx')
    if name == 'aliked':
        config['aliked_model_path'] = str(MODELS_DIR / 'aliked_lightglue.onnx')
    if component_index == 2 and config['backend'] == 'opencv':
        config['mode'] = 'simple'
    return config


def close_models(extractor, matcher):
    for component in (getattr(extractor, 'detector', None), getattr(extractor, 'descriptor', None),
                      matcher._matcher):
        if component is not None and hasattr(component, 'close'):
            component.close()


def test_all_registered_methods_are_covered():
    assert {detector for detector, _, _ in METHOD_CASES} == ALL_DETECTORS
    assert {descriptor for _, descriptor, _ in METHOD_CASES} == ALL_DESCRIPTORS
    assert {matcher for _, _, matcher in METHOD_CASES} == ALL_MATCHERS


@pytest.mark.parametrize('detector, descriptor, matcher', METHOD_CASES,
                         ids=[f'{detector}-{descriptor}-{matcher}' for detector, descriptor, matcher in METHOD_CASES])
def test_method_runs_in_panorama_pipeline(detector, descriptor, matcher):
    logger = MagicMock(spec=logging.Logger)
    parameters = {
        'detector': detector,
        'descriptor': descriptor,
        'matcher': matcher,
        'sequential_matching': True,
        'src_config': {
            'detector': component_config(detector, 0),
            'descriptor': component_config(descriptor, 1),
            'matcher': component_config(matcher, 2),
        },
    }
    images = [cv.imread(str(ROOT_DIR / 'test_data' / name))
              for name in ('box.png', 'box_in_scene.png')]
    assert all(image is not None for image in images)

    extractor = FeatureExtractor.create('pipeline', parameters, logger)
    matcher_wrapper = MatcherWrapper.create(matcher, parameters, logger)
    try:
        features = [extractor.extract(image) for image in images]
        matcher_wrapper.set_images(images)
        pairwise_matches = matcher_wrapper.apply2(features)
    finally:
        close_models(extractor, matcher_wrapper)

    assert len(features) == len(images)
    assert len(pairwise_matches) == len(images) ** 2
