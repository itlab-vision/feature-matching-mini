
TORCH_DETECTORS = {'superpoint', 'superpoint_lightglue', 'disk_lightglue', 'sift_lightglue', 'aliked_lightglue',
                   'doghardnet_lightglue', 'd2net', 'xfeat', 'r2d2', 'loftr', 'roma'}

TORCH_DESCRIPTORS = {'superpoint', 'superpoint_lightglue', 'disk_lightglue', 'sift_lightglue', 'aliked_lightglue',
                     'doghardnet_lightglue', 'd2net', 'xfeat', 'r2d2', 'loftr', 'roma', 'tfeat', 'hardnet'}

TORCH_MATCHERS = {'lightglue', 'superglue', 'loftr', 'roma'}

TORCH_PIPELINES = {'loftr', 'roma'}

OPENCV_DETECTORS = {'sift', 'orb', 'fast', 'akaze', 'brisk', 'kaze', 'gftt', 'mser', 'agast', 'blob', 'star',
                    'harrislaplace', 'msd', 'disk', 'aliked'}

OPENCV_DESCRIPTORS = {'sift', 'orb', 'akaze', 'brisk', 'kaze', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid',
                      'vgg', 'boostdesc', 'disk', 'aliked'}

OPENCV_MATCHERS = {'bf', 'flann', 'lightglue'}

OPENCV_DNN_DETECTORS = {'disk', 'aliked'}

OPENCV_DNN_DESCRIPTORS = {'disk', 'aliked'}

OPENCV_DNN_MATCHERS = {'lightglue'}

ONNX_DETECTORS = {'disk', 'aliked'}

ONNX_DESCRIPTORS = {'disk', 'aliked'}

ONNX_MATCHERS = {'lightglue'}
OPENCV_MATCHERS_MODE = {'simple', 'knn'}


TORCH_ALGORITHMS = TORCH_DETECTORS | TORCH_DESCRIPTORS | TORCH_MATCHERS | TORCH_PIPELINES
OPENCV_ALGORITHMS = OPENCV_DETECTORS | OPENCV_DESCRIPTORS | OPENCV_MATCHERS
ONNX_ALGORITHMS = ONNX_DETECTORS | ONNX_DESCRIPTORS | ONNX_MATCHERS

DNN_DETECTORS = TORCH_DETECTORS | OPENCV_DNN_DETECTORS | ONNX_DETECTORS
DNN_DESCRIPTORS = TORCH_DESCRIPTORS | OPENCV_DNN_DESCRIPTORS | ONNX_DESCRIPTORS
DNN_MATCHERS = TORCH_MATCHERS | OPENCV_DNN_MATCHERS | ONNX_MATCHERS
DNN_PIPELINES = TORCH_PIPELINES
DNN_ALGORITHMS = DNN_DETECTORS | DNN_DESCRIPTORS | DNN_MATCHERS | DNN_PIPELINES

ALL_DETECTORS = TORCH_DETECTORS | OPENCV_DETECTORS | ONNX_DETECTORS
ALL_DESCRIPTORS = TORCH_DESCRIPTORS | OPENCV_DESCRIPTORS | ONNX_DESCRIPTORS
ALL_MATCHERS = TORCH_MATCHERS | OPENCV_MATCHERS | ONNX_MATCHERS
ALL_PIPELINES = TORCH_PIPELINES
ALL_ALGORITHMS = ALL_DETECTORS | ALL_DESCRIPTORS | ALL_MATCHERS | ALL_PIPELINES

BACKEND_ALGORITHMS = {
    'opencv': (OPENCV_DETECTORS, OPENCV_DESCRIPTORS, OPENCV_MATCHERS),
    'torch': (TORCH_DETECTORS, TORCH_DESCRIPTORS, TORCH_MATCHERS),
    'onnx': (ONNX_DETECTORS, ONNX_DESCRIPTORS, ONNX_MATCHERS)
}

DETECTOR_DESCRIPTOR_COMPATIBILITY = {
    'sift': {'opencv': {
        'opencv': ['sift', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
        'torch': ['tfeat', 'hardnet'],
    }},
    'orb': {'opencv': {
        'opencv': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg',
                   'boostdesc'],
        'torch': ['tfeat', 'hardnet'],
    }},
    'fast': {'opencv': {
        'opencv': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg',
                   'boostdesc'],
        'torch': ['tfeat', 'hardnet'],
    }},
    'akaze': {'opencv': {
        'opencv': ['sift', 'orb', 'akaze', 'brisk', 'kaze', 'brief', 'freak', 'daisy', 'latch', 'beblid',
                   'teblid', 'vgg', 'boostdesc'],
        'torch': ['tfeat', 'hardnet'],
    }},
    'brisk': {'opencv': {
        'opencv': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg',
                   'boostdesc'],
        'torch': ['tfeat', 'hardnet'],
    }},
    'kaze': {'opencv': {
        'opencv': ['sift', 'orb', 'brisk', 'kaze', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid',
                   'vgg', 'boostdesc'],
        'torch': ['tfeat', 'hardnet'],
    }},
    'gftt': {'opencv': {
        'opencv': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg',
                   'boostdesc'],
        'torch': ['tfeat', 'hardnet'],
    }},
    'mser': {'opencv': {
        'opencv': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg',
                   'boostdesc'],
        'torch': ['tfeat', 'hardnet'],
    }},
    'agast': {'opencv': {
        'opencv': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg',
                   'boostdesc'],
        'torch': ['tfeat', 'hardnet'],
    }},
    'blob': {'opencv': {
        'opencv': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg',
                   'boostdesc'],
        'torch': ['tfeat', 'hardnet'],
    }},
    'star': {'opencv': {
        'opencv': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg',
                   'boostdesc'],
        'torch': ['tfeat', 'hardnet'],
    }},
    'harrislaplace': {'opencv': {
        'opencv': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg',
                   'boostdesc'],
        'torch': ['tfeat', 'hardnet'],
    }},
    'msd': {'opencv': {
        'opencv': ['orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
        'torch': ['tfeat', 'hardnet'],
    }},
    'superpoint': {'torch': {'torch': ['superpoint']}},
    'superpoint_lightglue': {
        'torch': {'torch': ['superpoint_lightglue']},
    },
    'disk_lightglue': {
        'torch': {'torch': ['disk_lightglue']},
    },
    'disk': {'opencv': {'opencv': ['disk']}, 'onnx': {'onnx': ['disk']}},
    'sift_lightglue': {'torch': {'torch': ['sift_lightglue']}},
    'aliked_lightglue': {'torch': {'torch': ['aliked_lightglue']}},
    'aliked': {'opencv': {'opencv': ['aliked']}, 'onnx': {'onnx': ['aliked']}},
    'doghardnet_lightglue': {'torch': {'torch': ['doghardnet_lightglue']}},
    'd2net': {
        'torch': {'torch': ['d2net']},
    },
    'xfeat': {
        'torch': {'torch': ['xfeat']},
    },
    'r2d2': {'torch': {'torch': ['r2d2']}},
    'loftr': {'torch': {'torch': ['loftr']}},
    'roma': {'torch': {'torch': ['roma']}},
}

DESCRIPTOR_MATCHER_COMPATIBILITY = {
    'sift': {'opencv': {'opencv': ['bf', 'flann']}},
    'orb': {'opencv': {'opencv': ['bf', 'flann']}},
    'akaze': {'opencv': {'opencv': ['bf', 'flann']}},
    'brisk': {'opencv': {'opencv': ['bf', 'flann']}},
    'kaze': {'opencv': {'opencv': ['bf', 'flann']}},
    'brief': {'opencv': {'opencv': ['bf', 'flann']}},
    'freak': {'opencv': {'opencv': ['bf', 'flann']}},
    'daisy': {'opencv': {'opencv': ['bf', 'flann']}},
    'latch': {'opencv': {'opencv': ['bf', 'flann']}},
    'beblid': {'opencv': {'opencv': ['bf', 'flann']}},
    'teblid': {'opencv': {'opencv': ['bf', 'flann']}},
    'vgg': {'opencv': {'opencv': ['bf', 'flann']}},
    'boostdesc': {'opencv': {'opencv': ['bf', 'flann']}},
    'superpoint': {'torch': {
        'opencv': ['bf', 'flann'],
        'torch': ['superglue'],
    }},
    'superpoint_lightglue': {
        'torch': {'torch': ['lightglue']},
    },
    'disk_lightglue': {
        'torch': {'torch': ['lightglue']},
    },
    'disk': {'opencv': {'opencv': ['bf', 'flann', 'lightglue']},
             'onnx': {'opencv': ['bf', 'flann'], 'onnx': ['lightglue']}},
    'sift_lightglue': {'torch': {'torch': ['lightglue']}},
    'aliked_lightglue': {'torch': {'torch': ['lightglue']}},
    'aliked': {'opencv': {'opencv': ['bf', 'flann', 'lightglue']},
               'onnx': {'opencv': ['bf', 'flann'], 'onnx': ['lightglue']}},
    'doghardnet_lightglue': {'torch': {'torch': ['lightglue']}},
    'xfeat': {
        'torch': {'opencv': ['bf', 'flann']},
    },
    'tfeat': {
        'torch': {'opencv': ['bf', 'flann']},
    },
    'hardnet': {
        'torch': {'opencv': ['bf', 'flann']},
    },
    'd2net': {
        'torch': {'opencv': ['bf', 'flann']},
    },
    'r2d2': {'torch': {'opencv': ['bf', 'flann']}},
    'loftr': {'torch': {'torch': ['loftr']}},
    'roma': {'torch': {'torch': ['roma']}},
}


def get_default_backend(algorithm_name, algorithm_index):
    if algorithm_name in BACKEND_ALGORITHMS['torch'][algorithm_index]:
        return 'torch'
    if algorithm_name in BACKEND_ALGORITHMS['opencv'][algorithm_index]:
        return 'opencv'
    if algorithm_name in BACKEND_ALGORITHMS['onnx'][algorithm_index]:
        return 'onnx'
    return None


def get_compatible_descriptors(detector, detector_backend, descriptor_backend):
    return DETECTOR_DESCRIPTOR_COMPATIBILITY.get(detector, {}).get(detector_backend, {}).get(descriptor_backend, [])


def get_compatible_matchers(descriptor, descriptor_backend, matcher_backend):
    return DESCRIPTOR_MATCHER_COMPATIBILITY.get(descriptor, {}).get(descriptor_backend, {}).get(matcher_backend, [])
