import cv2

EXPOS_COMP_CHOICES = {
    'gain_blocks': cv2.detail.ExposureCompensator_GAIN_BLOCKS,
    'gain': cv2.detail.ExposureCompensator_GAIN,
    'channel': cv2.detail.ExposureCompensator_CHANNELS,
    'channel_blocks': cv2.detail.ExposureCompensator_CHANNELS_BLOCKS,
    'no': cv2.detail.ExposureCompensator_NO,
}


BA_COST_CHOICES = {
    'ray': cv2.detail_BundleAdjusterRay,
    'reproj': cv2.detail_BundleAdjusterReproj,
    'affine': cv2.detail_BundleAdjusterAffinePartial,
    'no': cv2.detail_NoBundleAdjuster,
}


FEATURES_FIND_CHOICES = {
    name: creator for name, creator in {
        'orb': getattr(cv2, 'ORB_create', None),
        'sift': getattr(cv2, 'SIFT_create', None),
        'brisk': getattr(cv2, 'BRISK_create', None),
        'akaze': getattr(cv2, 'AKAZE_create', None),
    }.items() if creator is not None
}


ESTIMATOR_CHOICES = {
    'homography': cv2.detail_HomographyBasedEstimator,
    'affine': cv2.detail_AffineBasedEstimator,
}


WARP_CHOICES = (
    'spherical',
    'plane',
    'affine',
    'cylindrical',
    'fisheye',
    'stereographic',
    'compressedPlaneA2B1',
    'compressedPlaneA1.5B1',
    'compressedPlanePortraitA2B1',
    'compressedPlanePortraitA1.5B1',
    'paniniA2B1',
    'paniniA1.5B1',
    'paniniPortraitA2B1',
    'paniniPortraitA1.5B1',
    'mercator',
    'transverseMercator',
)


WAVE_CORRECT_CHOICES = {
    'horiz': cv2.detail.WAVE_CORRECT_HORIZ,
    'no': None,
    'vert': cv2.detail.WAVE_CORRECT_VERT,
}


BLEND_CHOICES = ('multiband', 'feather', 'no',)
