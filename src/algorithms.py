
DNN_DETECTORS = {'superpoint', 'superpoint_lightglue', 'disk_lightglue', 'sift_lightglue', 'aliked_lightglue',
                 'doghardnet_lightglue'}

DNN_DESCRIPTORS = {'superpoint', 'superpoint_lightglue', 'disk_lightglue', 'sift_lightglue', 'aliked_lightglue',
                   'doghardnet_lightglue'}

DNN_MATCHERS = {'lightglue'}

OPENCV_DETECTORS = {'sift', 'orb', 'fast', 'akaze', 'brisk', 'kaze', 'gftt', 'mser', 'agast', 'blob', 'star',
                    'harrislaplace', 'msd', 'diskopencv', 'alikedopencv'}

OPENCV_DESCRIPTORS = {'sift', 'orb', 'akaze', 'brisk', 'kaze', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid',
                      'vgg', 'boostdesc', 'diskopencv', 'alikedopencv', 'tfeat'}

OPENCV_MATCHERS = {'bf', 'flann', 'lightglueopencv'}
OPENCV_MATCHERS_MODE = {'simple', 'knn'}


DNN_ALGORITHMS = DNN_DETECTORS | DNN_DESCRIPTORS | DNN_MATCHERS
OPENCV_ALGORITHMS = OPENCV_DETECTORS | OPENCV_DESCRIPTORS | OPENCV_MATCHERS
ALL_ALGORITHMS = DNN_ALGORITHMS | OPENCV_ALGORITHMS

DETECTOR_DESCRIPTOR_COMPATIBILITY = {
    'sift': ['sift', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc', 'tfeat'],
    'orb': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc',
            'tfeat'],
    'fast': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc',
             'tfeat'],
    'akaze': ['sift', 'orb', 'akaze', 'brisk', 'kaze', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid',
              'vgg', 'boostdesc', 'tfeat'],
    'brisk': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc',
              'tfeat'],
    'kaze': ['sift', 'orb', 'brisk', 'kaze', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg',
             'boostdesc', 'tfeat'],
    'gftt': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc',
             'tfeat'],
    'mser': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc',
             'tfeat'],
    'agast': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc',
              'tfeat'],
    'blob': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc',
             'tfeat'],
    'star': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc',
             'tfeat'],
    'harrislaplace': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg',
                      'boostdesc', 'tfeat'],
    'msd': ['orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc', 'tfeat'],

    'superpoint': ['superpoint'],
    'superpoint_lightglue': ['superpoint_lightglue'],
    'disk_lightglue': ['disk_lightglue'],
    'diskopencv': ['diskopencv'],
    'sift_lightglue': ['sift_lightglue'],
    'aliked_lightglue': ['aliked_lightglue'],
    'alikedopencv': ['alikedopencv'],
    'doghardnet_lightglue': ['doghardnet_lightglue'],
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
    'superpoint': ['bf', 'flann'],
    'superpoint_lightglue': ['lightglue'],
    'disk_lightglue': ['lightglue'],
    'diskopencv': ['bf', 'flann', 'lightglueopencv'],
    'sift_lightglue': ['lightglue'],
    'aliked_lightglue': ['lightglue'],
    'alikedopencv': ['bf', 'flann', 'lightglueopencv'],
    'doghardnet_lightglue': ['lightglue'],
    'tfeat': ['bf', 'flann']
}
