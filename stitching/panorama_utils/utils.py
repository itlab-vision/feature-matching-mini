import cv2
import numpy as np


SRC_DESCRIPTOR_ONLY = {'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc',
                       'tfeat', 'hardnet'}
SRC_DETECTOR_ONLY = {'fast', 'kaze', 'gftt', 'mser', 'agast', 'blob', 'star', 'harrislaplace', 'msd'}
SRC_DETECTOR_DESCRIPTOR = {'d2net', 'xfeat', 'r2d2', 'disk_opencv', 'aliked_opencv',
                           'superpoint_lightglue', 'disk_lightglue', 'sift_lightglue',
                           'aliked_lightglue', 'doghardnet_lightglue'}
SRC_PIPELINES = {'loftr', 'roma'}

SRC_EXTRACTOR_NAMES = SRC_DESCRIPTOR_ONLY | SRC_DETECTOR_ONLY | SRC_DETECTOR_DESCRIPTOR | SRC_PIPELINES
SRC_MATCHER_NAMES = {'bf', 'flann', 'lightglue', 'lightglue_opencv', 'superglue', 'loftr', 'roma'}

DEFAULT_DETECTOR = 'sift'
DEFAULT_DESCRIPTOR = 'sift'


def import_src():
    """Import the src feature-matching library and return its base classes.

    Importing `src` registers every detector/descriptor/matcher implementation,
    so the Detector/Descriptor/Matcher.create() factories cover all methods.
    """
    try:
        from src.detectors import Detector
        from src.descriptors import Descriptor
        from src.matchers import Matcher
    except ImportError as e:
        raise RuntimeError(
            'The src feature-matching library is not available. '
            'Run inside the feature-matching-env environment (repo root on PYTHONPATH).'
        ) from e
    return Detector, Descriptor, Matcher


def src_config(parameters):
    cfg = parameters.get('src_config', {})
    return cfg if isinstance(cfg, dict) else {}


def resolve_src_names(parameters):
    """Return (detector_name, descriptor_name) resolved from stitching parameters.

    Primary config keys are `detector` and `descriptor`. Legacy keys
    `features` / `src_detector` / `src_descriptor` are still supported.
    """
    features = parameters.get('features', 'pipeline')

    detector = parameters.get('detector')
    descriptor = parameters.get('descriptor')
    if detector is not None or descriptor is not None:
        detector = detector or parameters.get('src_detector', DEFAULT_DETECTOR)
        descriptor = descriptor or parameters.get('src_descriptor', detector)
        return detector, descriptor

    if features == 'pipeline':
        detector = parameters.get('src_detector', DEFAULT_DETECTOR)
        descriptor = parameters.get('src_descriptor', detector)
        return detector, descriptor

    if features in SRC_DESCRIPTOR_ONLY:
        return parameters.get('src_detector', DEFAULT_DETECTOR), features

    if features in SRC_DETECTOR_ONLY:
        return features, parameters.get('src_descriptor', DEFAULT_DESCRIPTOR)

    return features, features


def resolve_src_matcher(parameters):
    matcher = parameters.get('matcher', 'pipeline')
    if matcher == 'pipeline':
        return parameters.get('src_matcher', 'bf')
    return matcher


def validate_detector_descriptor(detector_name, descriptor_name):
    from src.algorithms import DETECTOR_DESCRIPTOR_COMPATIBILITY
    allowed = DETECTOR_DESCRIPTOR_COMPATIBILITY.get(detector_name, [])
    if isinstance(allowed, dict):
        allowed = [d for backends in allowed.values()
                   for descriptors in backends.values() for d in descriptors]
    if descriptor_name not in allowed:
        raise ValueError(
            f'Descriptor {descriptor_name!r} is not compatible with detector {detector_name!r}. '
            f'Allowed: {allowed}'
        )


def validate_descriptor_matcher(descriptor_name, matcher_name):
    from src.algorithms import DESCRIPTOR_MATCHER_COMPATIBILITY
    allowed = DESCRIPTOR_MATCHER_COMPATIBILITY.get(descriptor_name, [])
    if isinstance(allowed, dict):
        allowed = [m for backends in allowed.values()
                   for matchers in backends.values() for m in matchers]
    if matcher_name not in allowed:
        raise ValueError(
            f'Matcher {matcher_name!r} is not compatible with descriptor {descriptor_name!r}. '
            f'Allowed: {allowed}'
        )


def as_ndarray(data):
    if isinstance(data, cv2.UMat):
        return data.get()
    return data


def src_features_to_image_features(features, image, img_idx=80):
    """Convert a src-style features dict to cv2.detail.ImageFeatures."""
    img_h, img_w = image.shape[:2]
    features_cv = cv2.detail.ImageFeatures()
    features_cv.img_idx = img_idx
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
        data['width'] = width
        data['height'] = height
    return data
