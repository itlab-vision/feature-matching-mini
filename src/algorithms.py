
DNN_DETECTORS = {'superpoint', 'superpoint_lightglue', 'disk_lightglue', 'sift_lightglue', 'aliked_lightglue',
                 'doghardnet_lightglue', 'd2net', 'r2d2', 'loftr', 'efficientloftr', 'roma'}

DNN_DESCRIPTORS = {'superpoint', 'superpoint_lightglue', 'disk_lightglue', 'sift_lightglue', 'aliked_lightglue',
                   'doghardnet_lightglue', 'd2net', 'r2d2', 'loftr', 'efficientloftr', 'roma'}

DNN_MATCHERS = {'lightglue', 'superglue', 'loftr', 'efficientloftr', 'roma'}

DNN_PIPELINES = {'loftr', 'roma'}

OPENCV_DETECTORS = {'sift', 'orb', 'fast', 'akaze', 'brisk', 'kaze', 'gftt', 'mser', 'agast', 'blob', 'star',
                    'harrislaplace', 'msd', 'diskopencv', 'alikedopencv'}

OPENCV_DESCRIPTORS = {'sift', 'orb', 'akaze', 'brisk', 'kaze', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid',
                      'vgg', 'boostdesc', 'diskopencv', 'alikedopencv'}

OPENCV_MATCHERS = {'bf', 'flann'}
OPENCV_MATCHERS_MODE = {'simple', 'knn'}


DNN_ALGORITHMS = DNN_DETECTORS | DNN_DESCRIPTORS | DNN_MATCHERS
OPENCV_ALGORITHMS = OPENCV_DETECTORS | OPENCV_DESCRIPTORS | OPENCV_MATCHERS
ALL_ALGORITHMS = DNN_ALGORITHMS | OPENCV_ALGORITHMS

DETECTOR_DESCRIPTOR_COMPATIBILITY = {
    'sift': ['sift', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
    'orb': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
    'fast': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
    'akaze': ['sift', 'orb', 'akaze', 'brisk', 'kaze', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid',
              'vgg', 'boostdesc'],
    'brisk': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
    'kaze': ['sift', 'orb', 'brisk', 'kaze', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg',
             'boostdesc'],
    'gftt': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
    'mser': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
    'agast': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
    'blob': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
    'star': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
    'harrislaplace': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg',
                      'boostdesc'],
    'msd': ['orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],

    'superpoint': ['superpoint'],
    'superpoint_lightglue': ['superpoint_lightglue'],
    'disk_lightglue': ['disk_lightglue'],
    'diskopencv': ['diskopencv'],
    'sift_lightglue': ['sift_lightglue'],
    'aliked_lightglue': ['aliked_lightglue'],
    'alikedopencv': ['alikedopencv'],
    'doghardnet_lightglue': ['doghardnet_lightglue'],
    'd2net': ['d2net'],
    'r2d2': ['r2d2'],
    'loftr': ['loftr'],
    'efficientloftr': ['efficientloftr'],
    'roma': ['roma'],
}

DESCRIPTOR_MATCHER_COMPATIBILITY = {
    'sift': ['bf', 'flann'],
    'orb': ['bf', 'flann'],
    'akaze': ['bf', 'flann'],
    'brisk': ['bf', 'flann'],
    'kaze': ['bf', 'flann'],
    'brief': ['bf', 'flann'],
    'freak': ['bf', 'flann'],
    'daisy': ['bf', 'flann'],
    'latch': ['bf', 'flann'],
    'beblid': ['bf', 'flann'],
    'teblid': ['bf', 'flann'],
    'vgg': ['bf', 'flann'],
    'boostdesc': ['bf', 'flann'],

    'superpoint': ['bf', 'flann', 'superglue'],
    'superpoint_lightglue': ['lightglue'],
    'disk_lightglue': ['lightglue'],
    'diskopencv': ['bf', 'flann'],
    'sift_lightglue': ['lightglue'],
    'aliked_lightglue': ['lightglue'],
    'alikedopencv': ['bf', 'flann'],
    'doghardnet_lightglue': ['lightglue'],
    'd2net': ['bf', 'flann'],
    'r2d2': ['bf', 'flann'],
    'loftr': ['loftr'],
    'efficientloftr': ['efficientloftr'],
    'roma': ['roma'],
}
