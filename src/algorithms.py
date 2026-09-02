
DNN_DETECTORS = {'superpoint', 'superpoint_lightglue', 'disk_lightglue', 'sift_lightglue', 'aliked_lightglue',
                 'doghardnet_lightglue', 'd2net', 'xfeat', 'r2d2', 'loftr', 'roma'}

DNN_DESCRIPTORS = {'superpoint', 'superpoint_lightglue', 'disk_lightglue', 'sift_lightglue', 'aliked_lightglue',
                   'doghardnet_lightglue', 'd2net', 'xfeat', 'r2d2', 'loftr', 'roma'}

DNN_MATCHERS = {'lightglue', 'superglue', 'loftr', 'roma'}

DNN_PIPELINES = {'loftr', 'roma'}

OPENCV_DETECTORS = {'sift', 'orb', 'fast', 'akaze', 'brisk', 'kaze', 'gftt', 'mser', 'agast', 'blob', 'star',
                    'harrislaplace', 'msd', 'diskopencv', 'alikedopencv'}

OPENCV_DESCRIPTORS = {'sift', 'orb', 'akaze', 'brisk', 'kaze', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid',
                      'vgg', 'boostdesc', 'diskopencv', 'alikedopencv', 'tfeat', 'hardnet'}

OPENCV_MATCHERS = {'bf', 'flann', 'lightglueopencv'}
OPENCV_MATCHERS_MODE = {'simple', 'knn'}


DNN_ALGORITHMS = DNN_DETECTORS | DNN_DESCRIPTORS | DNN_MATCHERS
OPENCV_ALGORITHMS = OPENCV_DETECTORS | OPENCV_DESCRIPTORS | OPENCV_MATCHERS

ALL_DETECTORS = DNN_DETECTORS | OPENCV_DETECTORS
ALL_DESCRIPTORS = DNN_DESCRIPTORS | OPENCV_DESCRIPTORS
ALL_MATCHERS = DNN_MATCHERS | OPENCV_MATCHERS
ALL_ALGORITHMS = ALL_DETECTORS | ALL_DESCRIPTORS | ALL_MATCHERS

DETECTOR_DESCRIPTOR_COMPATIBILITY = {
    'sift': ['sift', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc', 'tfeat',
             'hardnet'],
    'orb': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc',
            'tfeat', 'hardnet'],
    'fast': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc',
             'tfeat', 'hardnet'],
    'akaze': ['sift', 'orb', 'akaze', 'brisk', 'kaze', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid',
              'vgg', 'boostdesc', 'tfeat', 'hardnet'],
    'brisk': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc',
              'tfeat', 'hardnet'],
    'kaze': ['sift', 'orb', 'brisk', 'kaze', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg',
             'boostdesc', 'tfeat', 'hardnet'],
    'gftt': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc',
             'tfeat', 'hardnet'],
    'mser': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc',
             'tfeat', 'hardnet'],
    'agast': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc',
              'tfeat', 'hardnet'],
    'blob': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc',
             'tfeat', 'hardnet'],
    'star': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc',
             'tfeat', 'hardnet'],
    'harrislaplace': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg',
                      'boostdesc', 'tfeat', 'hardnet'],
    'msd': ['orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc', 'tfeat',
            'hardnet'],

    'superpoint': ['superpoint'],
    'superpoint_lightglue': ['superpoint_lightglue'],
    'disk_lightglue': ['disk_lightglue'],
    'diskopencv': ['diskopencv'],
    'sift_lightglue': ['sift_lightglue'],
    'aliked_lightglue': ['aliked_lightglue'],
    'alikedopencv': ['alikedopencv'],
    'doghardnet_lightglue': ['doghardnet_lightglue'],
    'd2net': ['d2net'],
    'xfeat': ['xfeat'],
    'r2d2': ['r2d2'],
    'loftr': ['loftr'],
    # 'roma': ['roma'],
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
    'diskopencv': ['bf', 'flann', 'lightglueopencv'],
    'sift_lightglue': ['lightglue'],
    'aliked_lightglue': ['lightglue'],
    'alikedopencv': ['bf', 'flann', 'lightglueopencv'],
    'doghardnet_lightglue': ['lightglue'],
    'xfeat': ['bf', 'flann'],
    'tfeat': ['bf', 'flann'],
    'hardnet': ['bf', 'flann'],
    'd2net': ['bf', 'flann'],
    'r2d2': ['bf', 'flann'],
    'loftr': ['loftr'],
    # 'roma': ['roma'],
}
