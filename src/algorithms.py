
TORCH_DETECTORS = {'superpoint', 'superpoint_lightglue', 'disk_lightglue', 'sift_lightglue', 'aliked_lightglue',
                 'doghardnet_lightglue', 'd2net', 'xfeat', 'r2d2', 'loftr', 'roma'}

TORCH_DESCRIPTORS = {'superpoint', 'superpoint_lightglue', 'disk_lightglue', 'sift_lightglue', 'aliked_lightglue',
                   'doghardnet_lightglue', 'd2net', 'xfeat', 'r2d2', 'loftr', 'roma'}

TORCH_MATCHERS = {'lightglue', 'superglue', 'loftr', 'roma'}

TORCH_PIPELINES = {'loftr', 'roma'}

OPENCV_DETECTORS = {'sift', 'orb', 'fast', 'akaze', 'brisk', 'kaze', 'gftt', 'mser', 'agast', 'blob', 'star',
                    'harrislaplace', 'msd', 'disk_opencv', 'aliked_opencv'}

OPENCV_DESCRIPTORS = {'sift', 'orb', 'akaze', 'brisk', 'kaze', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid',
                      'vgg', 'boostdesc', 'disk_opencv', 'aliked_opencv', 'tfeat', 'hardnet'}

OPENCV_MATCHERS = {'bf', 'flann', 'lightglue_opencv'}
OPENCV_MATCHERS_MODE = {'simple', 'knn'}

EXECUTORCH_DETECTORS = {'superpoint_lightglue_executorch', 'disk_lightglue_executorch', 'd2net_executorch',
                            'xfeat_executorch'}
EXECUTORCH_DESCRIPTORS = {'superpoint_lightglue_executorch', 'disk_lightglue_executorch', 'd2net_executorch',
                              'xfeat_executorch', 'tfeat_executorch', 'hardnet_executorch'}
EXECUTORCH_MATCHERS = {'lightglue_executorch', 'superglue_executorch'}

TORCH_ALGORITHMS = TORCH_DETECTORS | TORCH_DESCRIPTORS | TORCH_MATCHERS | TORCH_PIPELINES
OPENCV_ALGORITHMS = OPENCV_DETECTORS | OPENCV_DESCRIPTORS | OPENCV_MATCHERS
EXECUTORCH_ALGORITHMS = EXECUTORCH_DETECTORS | EXECUTORCH_DESCRIPTORS | EXECUTORCH_MATCHERS

DNN_DETECTORS = TORCH_DETECTORS | EXECUTORCH_DETECTORS
DNN_DESCRIPTORS = TORCH_DESCRIPTORS | EXECUTORCH_DETECTORS
DNN_MATCHERS = TORCH_MATCHERS | EXECUTORCH_MATCHERS
DNN_PIPELINES = TORCH_PIPELINES
DNN_ALGORITHMS = DNN_DETECTORS | DNN_DESCRIPTORS | DNN_MATCHERS | DNN_PIPELINES

ALL_DETECTORS = TORCH_DETECTORS | OPENCV_DETECTORS | EXECUTORCH_DETECTORS
ALL_DESCRIPTORS = TORCH_DESCRIPTORS | OPENCV_DESCRIPTORS | EXECUTORCH_DESCRIPTORS
ALL_MATCHERS = TORCH_MATCHERS | OPENCV_MATCHERS | EXECUTORCH_MATCHERS
ALL_PIPELINES = TORCH_PIPELINES
ALL_ALGORITHMS = ALL_DETECTORS | ALL_DESCRIPTORS | ALL_MATCHERS | ALL_PIPELINES

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
    'disk_opencv': ['disk_opencv'],
    'sift_lightglue': ['sift_lightglue'],
    'aliked_lightglue': ['aliked_lightglue'],
    'aliked_opencv': ['aliked_opencv'],
    'doghardnet_lightglue': ['doghardnet_lightglue'],
    'd2net': ['d2net'],
    'xfeat': ['xfeat'],
    'r2d2': ['r2d2'],
    'loftr': ['loftr'],
    'roma': ['roma'],
    'superpoint_lightglue_executorch': ['superpoint_lightglue_executorch'],
    'disk_lightglue_executorch': ['disk_lightglue_executorch'],
    'd2net_executorch': ['d2net_executorch'],
    'xfeat_executorch': ['xfeat_executorch'],
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
    'disk_opencv': ['bf', 'flann', 'lightglue_opencv'],
    'sift_lightglue': ['lightglue'],
    'aliked_lightglue': ['lightglue'],
    'aliked_opencv': ['bf', 'flann', 'lightglue_opencv'],
    'doghardnet_lightglue': ['lightglue'],
    'xfeat': ['bf', 'flann'],
    'tfeat': ['bf', 'flann'],
    'hardnet': ['bf', 'flann'],
    'd2net': ['bf', 'flann'],
    'r2d2': ['bf', 'flann'],
    'loftr': ['loftr'],
    'roma': ['roma'],
    'superpoint_lightglue_executorch': ['lightglue_executorch'],
    'disk_lightglue_executorch': ['lightglue_executorch'],
    'd2net_executorch': ['bf', 'flann'],
    'tfeat_executorch': ['bf', 'flann'],
    'hardnet_executorch': ['bf', 'flann'],
    'xfeat_executorch': ['bf', 'flann'],
}
