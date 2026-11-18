from importlib import import_module

import cv2
import numpy as np

from src.algorithms import ALL_DESCRIPTORS, ALL_DETECTORS


def import_src(*backends):
    """Load the src factories and register implementations for the selected backends."""
    from src.detectors import Detector
    from src.descriptors import Descriptor
    from src.matchers import Matcher

    import src.dnn_descriptor  # noqa: F401
    import src.dnn_extractors  # noqa: F401
    import src.dnn_matchers  # noqa: F401
    import src.dnn_pipeline  # noqa: F401
    import src.backend.io_adapter  # noqa: F401

    for backend in dict.fromkeys(backends):
        import_module(f'src.backend.{backend}')
    return Detector, Descriptor, Matcher


def src_stage_config(parameters, stage, name):
    from src.algorithms import BACKEND_ALGORITHMS, get_default_backend

    index = ('detector', 'descriptor', 'matcher').index(stage)
    config = dict((parameters.get('src_config') or {}).get(stage) or {})
    legacy_opencv = f'{name}_opencv' in (
        parameters.get(stage), parameters.get(f'src_{stage}'), parameters.get('features')
    )
    backend = config.get('backend') or ('opencv' if legacy_opencv else get_default_backend(name, index))
    backend = backend.lower() if backend else None
    if backend not in BACKEND_ALGORITHMS or name not in BACKEND_ALGORITHMS[backend][index]:
        raise ValueError(f'{stage.capitalize()} {name!r} is unavailable on backend {backend!r}')
    config['backend'] = backend
    return config


def _src_name(name):
    return name.lower().removesuffix('_opencv')


def resolve_src_names(parameters):
    """Return (detector_name, descriptor_name) resolved from stitching parameters.

    Primary config keys are `detector` and `descriptor`. Legacy keys
    `features` / `src_detector` / `src_descriptor` are still supported.
    """
    features = _src_name(parameters.get('features', 'pipeline'))
    detector = parameters.get('detector')
    descriptor = parameters.get('descriptor')

    if detector is not None or descriptor is not None or features == 'pipeline':
        detector = detector or parameters.get('src_detector', 'sift')
        descriptor = descriptor or parameters.get('src_descriptor', detector)
        return _src_name(detector), _src_name(descriptor)

    if features == 'superpoint' and resolve_src_matcher(parameters) == 'lightglue':
        return 'superpoint_lightglue', 'superpoint_lightglue'
    if features in ALL_DESCRIPTORS and features not in ALL_DETECTORS:
        return _src_name(parameters.get('src_detector', 'sift')), features
    if features in ALL_DETECTORS and features not in ALL_DESCRIPTORS:
        return features, _src_name(parameters.get('src_descriptor', 'sift'))
    return features, features


def resolve_src_matcher(parameters):
    matcher = parameters.get('matcher', 'pipeline')
    if matcher == 'pipeline':
        return _src_name(parameters.get('src_matcher', 'bf'))
    return _src_name(matcher)


def validate_detector_descriptor(detector, descriptor, detector_backend, descriptor_backend):
    from src.algorithms import get_compatible_descriptors

    allowed = get_compatible_descriptors(detector, detector_backend, descriptor_backend)
    if descriptor not in allowed:
        raise ValueError(
            f'Descriptor {descriptor!r} ({descriptor_backend}) is not compatible with '
            f'detector {detector!r} ({detector_backend}). Allowed: {allowed}'
        )


def validate_descriptor_matcher(descriptor, matcher, descriptor_backend, matcher_backend):
    from src.algorithms import get_compatible_matchers

    allowed = get_compatible_matchers(descriptor, descriptor_backend, matcher_backend)
    if matcher not in allowed:
        raise ValueError(
            f'Matcher {matcher!r} ({matcher_backend}) is not compatible with '
            f'descriptor {descriptor!r} ({descriptor_backend}). Allowed: {allowed}'
        )


def as_ndarray(data):
    if isinstance(data, cv2.UMat):
        return data.get()
    return data


def src_features_to_image_features(features, image):
    """Convert a src-style features dict to cv2.detail.ImageFeatures."""
    img_h, img_w = image.shape[:2]
    features_cv = cv2.detail.ImageFeatures()
    features_cv.img_size = (img_w, img_h)

    keypoints = features.get('keypoints')
    if keypoints is None:
        keypoints = ()
    scores = features.get('scores')
    cv_keypoints = []
    if len(keypoints) > 0:
        if isinstance(keypoints[0], cv2.KeyPoint):
            cv_keypoints = list(keypoints)
        else:
            keypoints_np = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
            for i, (x, y) in enumerate(keypoints_np):
                score = float(scores[i]) if scores is not None and i < len(scores) else 0.0
                cv_keypoints.append(cv2.KeyPoint(float(x), float(y), 1.0, -1, score, 0, i))
    features_cv.keypoints = tuple(cv_keypoints)

    descriptors = as_ndarray(features.get('descriptors'))
    if descriptors is not None and len(descriptors) > 0:
        descriptors = np.asarray(descriptors)
        if (descriptors.ndim == 2 and descriptors.shape[0] != len(cv_keypoints)
                and descriptors.shape[1] == len(cv_keypoints)):
            descriptors = descriptors.T
        features_cv.descriptors = np.ascontiguousarray(descriptors)
    else:
        features_cv.descriptors = np.empty((0, 0), dtype=np.float32)
    return features_cv


def image_features_to_src_dict(features_cv):
    """Convert cv2.detail.ImageFeatures to a src-style features dict."""
    keypoints = features_cv.keypoints or ()
    keypoints_cv = list(keypoints)
    scores = np.asarray([kp.response for kp in keypoints], dtype=np.float32)
    descriptors = as_ndarray(features_cv.descriptors)
    descriptors_np = np.asarray(descriptors) if descriptors is not None else np.empty((0, 0), dtype=np.float32)
    data = {
        'keypoints': keypoints_cv,
        'descriptors': descriptors_np,
        'scores': scores,
    }
    if features_cv.img_size is not None:
        width, height = features_cv.img_size
        data.update(width=width, height=height, image_size=np.array([[width, height]]))
    return data
