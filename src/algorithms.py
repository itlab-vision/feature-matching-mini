
DNN_DETECTORS = {'superpoint', 'superpoint_lightglue', 'disk_lightglue', 'sift_lightglue', 'aliked_lightglue',
                 'doghardnet_lightglue', 'xfeat', 'd2net'}

DNN_DESCRIPTORS = {'superpoint', 'superpoint_lightglue', 'disk_lightglue', 'sift_lightglue', 'aliked_lightglue',
                   'doghardnet_lightglue', 'xfeat', 'd2net'}

DNN_MATCHERS = {'lightglue', 'superglue'}

OPENCV_DETECTORS = {'sift', 'orb', 'fast', 'akaze', 'brisk', 'kaze', 'gftt', 'mser', 'agast', 'blob', 'star',
                    'harrislaplace', 'msd', 'diskopencv', 'alikedopencv'}

OPENCV_DESCRIPTORS = {'sift', 'orb', 'akaze', 'brisk', 'kaze', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid',
                      'vgg', 'boostdesc', 'diskopencv', 'alikedopencv', 'tfeat', 'hardnet'}

OPENCV_MATCHERS = {'bf', 'flann', 'lightglueopencv'}
OPENCV_MATCHERS_MODE = {'simple', 'knn'}

DNN_EXECUTORCH_DETECTORS = {'superpoint_lightglue_executorch', 'disk_lightglue_executorch', 'd2net_executorch'}
DNN_EXECUTORCH_DESCRIPTORS = {'superpoint_lightglue_executorch', 'disk_lightglue_executorch', 'd2net_executorch'}
DNN_EXECUTORCH_MATCHERS = {'lightglue_executorch', 'superglue_executorch'}

OPENCV_EXECUTORCH_DESCRIPTORS = {'tfeat_executorch', 'hardnet_executorch'}

DNN_ALGORITHMS = (DNN_DETECTORS | DNN_DESCRIPTORS | DNN_MATCHERS
                  | DNN_EXECUTORCH_DETECTORS | DNN_EXECUTORCH_DESCRIPTORS | DNN_EXECUTORCH_MATCHERS)
OPENCV_ALGORITHMS = OPENCV_DETECTORS | OPENCV_DESCRIPTORS | OPENCV_MATCHERS | OPENCV_EXECUTORCH_DESCRIPTORS
EXECUTORCH_ALGORITHMS = (DNN_EXECUTORCH_DETECTORS | DNN_EXECUTORCH_DESCRIPTORS | DNN_EXECUTORCH_MATCHERS
                         | OPENCV_EXECUTORCH_DESCRIPTORS)

ALL_DETECTORS = DNN_DETECTORS | OPENCV_DETECTORS | DNN_EXECUTORCH_DETECTORS
ALL_DESCRIPTORS = DNN_DESCRIPTORS | OPENCV_DESCRIPTORS | DNN_EXECUTORCH_DESCRIPTORS | OPENCV_EXECUTORCH_DESCRIPTORS
ALL_MATCHERS = DNN_MATCHERS | OPENCV_MATCHERS | DNN_EXECUTORCH_MATCHERS
ALL_ALGORITHMS = ALL_DETECTORS | ALL_DESCRIPTORS | ALL_MATCHERS

DETECTOR_DESCRIPTOR_COMPATIBILITY = {
    'sift': ['sift', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc', 'tfeat',
             'hardnet', 'tfeat_executorch', 'hardnet_executorch'],
    'orb': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc',
            'tfeat', 'hardnet', 'tfeat_executorch', 'hardnet_executorch'],
    'fast': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc',
             'tfeat', 'hardnet', 'tfeat_executorch', 'hardnet_executorch'],
    'akaze': ['sift', 'orb', 'akaze', 'brisk', 'kaze', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid',
              'vgg', 'boostdesc', 'tfeat', 'hardnet', 'tfeat_executorch', 'hardnet_executorch'],
    'brisk': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc',
              'tfeat', 'hardnet', 'tfeat_executorch', 'hardnet_executorch'],
    'kaze': ['sift', 'orb', 'brisk', 'kaze', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg',
             'boostdesc', 'tfeat', 'hardnet', 'tfeat_executorch', 'hardnet_executorch'],
    'gftt': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc',
             'tfeat', 'hardnet', 'tfeat_executorch', 'hardnet_executorch'],
    'mser': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc',
             'tfeat', 'hardnet', 'tfeat_executorch', 'hardnet_executorch'],
    'agast': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc',
              'tfeat', 'hardnet', 'tfeat_executorch', 'hardnet_executorch'],
    'blob': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc',
             'tfeat', 'hardnet', 'tfeat_executorch', 'hardnet_executorch'],
    'star': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc',
             'tfeat', 'hardnet', 'tfeat_executorch', 'hardnet_executorch'],
    'harrislaplace': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg',
                      'boostdesc', 'tfeat', 'hardnet', 'tfeat_executorch', 'hardnet_executorch'],
    'msd': ['orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc', 'tfeat',
            'hardnet', 'tfeat_executorch', 'hardnet_executorch'],

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
    'superpoint_lightglue_executorch': ['superpoint_lightglue_executorch'],
    'disk_lightglue_executorch': ['disk_lightglue_executorch'],
    'd2net_executorch': ['d2net_executorch'],
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
    'superpoint_lightglue_executorch': ['lightglue_executorch'],
    'disk_lightglue_executorch': ['lightglue_executorch'],
    'd2net_executorch': ['bf', 'flann'],
    'tfeat_executorch': ['bf', 'flann'],
    'hardnet_executorch': ['bf', 'flann'],
}
