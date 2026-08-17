import sys
import argparse
from pathlib import Path

from feature_matching.detectors import Detector  # noqa: E402
from feature_matching.descriptors import Descriptor  # noqa: E402
from feature_matching.matchers import Matcher, OpenCVMatcher  # noqa: E402
from feature_matching.algorithms import DNN_DETECTORS, DNN_DESCRIPTORS, DNN_MATCHERS, OPENCV_MATCHERS  # noqa: E402
from feature_matching.super_point import SuperPoint  # noqa: F401, E402
from feature_matching.lightglue_matcher import LightGlue  # noqa: F401, E402
from feature_matching.lightglue_pipeline import LightGlueFeatureExtractor  # noqa: F401, E402


def build_detector_config(args):
    config = dict()
    if (args.device is not None) and (args.detector in DNN_DETECTORS):
        config['device'] = args.device
    if args.det_nfeatures is not None:
        config['nfeatures'] = args.det_nfeatures
    if args.det_noctave is not None:
        config['nOctaveLayers'] = args.det_noctave
    if args.det_threshold is not None:
        config['threshold'] = args.det_threshold
    if args.disk_model_path is not None:
        config['disk_model_path'] = args.disk_model_path
    if args.aliked_model_path is not None:
        config['aliked_model_path'] = args.aliked_model_path
    return config


def build_descriptor_config(args):
    config = dict()
    if (args.device is not None) and (args.descriptor in DNN_DESCRIPTORS or args.descriptor in ['tfeat', 'hardnet']):
        config['device'] = args.device
    if args.des_nfeatures is not None:
        config['nfeatures'] = args.des_nfeatures
    if args.des_threshold is not None:
        config['threshold'] = args.des_threshold
    if args.des_scale is not None:
        config['scale_factor'] = args.des_scale
    if args.disk_model_path is not None:
        config['disk_model_path'] = args.disk_model_path
    if args.aliked_model_path is not None:
        config['aliked_model_path'] = args.aliked_model_path
    if args.tfeat_model_path is not None:
        config['tfeat_model_path'] = args.tfeat_model_path
    if args.hardnet_model_path is not None:
        config['hardnet_model_path'] = args.hardnet_model_path
    if args.patch_size is not None:
        config['patch_size'] = args.patch_size
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.magfactor is not None:
        config['magfactor'] = args.magfactor
    return config


def build_matcher_config(args):
    config = dict()
    if (args.device is not None) and (args.matcher in DNN_MATCHERS):
        config['device'] = args.device
    if args.matcher in OPENCV_MATCHERS:
        config['mode'] = args.matcher_mode
    if (args.k_knn is not None) and (args.matcher in OPENCV_MATCHERS):
        config['k'] = args.k_knn
    if args.mat_ratio is not None:
        config['ratio'] = args.mat_ratio
    if args.mat_cross_check is not None:
        config['cross_check'] = args.mat_cross_check
    if args.mat_score_threshold is not None:
        config['score_threshold'] = args.mat_score_threshold
    if args.lightglue_model_path is not None:
        config['lightglue_model_path'] = args.lightglue_model_path
    return config


def build_preprocessor_config(args):
    config = dict()
    if args.device is not None:
        config['device'] = args.device

    return config


def build_feature_matcher_config(args):
    return {
        'detector': build_detector_config(args),
        'descriptor': build_descriptor_config(args),
        'matcher': build_matcher_config(args),
        'preprocessor': build_preprocessor_config(args),
    }


def performance_tests_parser():
    arg_parser = argparse.ArgumentParser(
        description="Feature matching staged performance test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    available_detectors = list(Detector._METHODS.keys())
    available_descriptors = list(Descriptor._METHODS.keys())
    available_matchers = list(Matcher._METHODS.keys())
    available_matchers_modes = list(OpenCVMatcher._MODE)
    available_devices = ['cpu', 'cuda', 'mps']

    arg_parser.add_argument('-det', '--detector', type=str, default='sift',
                            choices=available_detectors, help='Detector algorithm')
    arg_parser.add_argument('-des', '--descriptor', type=str, default='sift',
                            choices=available_descriptors, help='Descriptor algorithm')
    arg_parser.add_argument('-mat', '--matcher', type=str, default='bf',
                            choices=available_matchers, help='Matching algorithm')

    arg_parser.add_argument('-i1', '--image1', type=Path, required=True,
                            help='Path to the first image')
    arg_parser.add_argument('-i2', '--image2', type=Path, required=True,
                            help='Path to the second image')

    arg_parser.add_argument('-d', '--device', type=str, default=None,
                            choices=available_devices, help='The device on which the script will be run')

    det_group = arg_parser.add_argument_group('Detector config')
    det_group.add_argument('-dn', '--det-nfeatures', type=int, default=None,
                           help='Max number of features to detect')
    det_group.add_argument('-do', '--det-noctave', type=int, default=None,
                           help='Number of octave layers')
    det_group.add_argument('-dt', '--det-threshold', type=float, default=None,
                           help='Detection threshold')
    det_group.add_argument('-mpd', '--disk-model-path', type=str, default=None,
                           help='Path to DISK opencv model')
    det_group.add_argument('-mpa', '--aliked-model-path', type=str, default=None,
                           help='Path to ALIKED opencv model')

    des_group = arg_parser.add_argument_group('Descriptor config')
    des_group.add_argument('-dsen', '--des-nfeatures', type=int, default=None,
                           help='Max number of features for descriptor')
    des_group.add_argument('-dsdt', '--des-threshold', type=float, default=None,
                           help='Descriptor threshold')
    des_group.add_argument('-dss', '--des-scale', type=float, default=None,
                           help='Scale factor')
    des_group.add_argument('-mpt', '--tfeat-model-path', type=str, default=None,
                           help='Path to TFeat model')
    des_group.add_argument('-mphn', '--hardnet-model-path', type=str, default=None,
                           help='Path to HardNet model')
    des_group.add_argument('-psize', '--patch-size', type=int, default=None,
                           help='Patch size for HardNet model')
    des_group.add_argument('-bsize', '--batch-size', type=int, default=None,
                           help='Batch size for HardNet model')
    des_group.add_argument('-mag', '--magfactor', type=int, default=None,
                           help='MagFactor for TFeat')

    mat_group = arg_parser.add_argument_group('Matcher config')
    mat_group.add_argument('-mat_m', '--matcher_mode', type=str, default='simple',
                           choices=available_matchers_modes, help='Matching mode')
    mat_group.add_argument('-k', '--k_knn', type=int, default=None,
                           help='K for knn mode')
    mat_group.add_argument('-mr', '--mat-ratio', type=float, default=None,
                           help='Ratio threshold for KNN')
    mat_group.add_argument('-mc', '--mat-cross-check', action='store_true', default=None,
                           help='Enable cross-check for BF matcher')
    mat_group.add_argument('-mst', '--mat-score-threshold', type=float, default=None,
                           help='scoreThreshold for LightGlue matcher')
    mat_group.add_argument('-mplg', '--lightglue-model-path', type=str, default=None,
                           help='Path to LightGlue opencv model')

    arg_parser.add_argument('-n', '--iterations', type=int, default=10,
                            help='Number of iterations for performance testing (default: 10)')
    return arg_parser.parse_args()


def build_hpatches_dataset_config(args):
    config = dict()
    if args.path is not None:
        config['path'] = args.path
    if args.num_scenes is not None:
        config['num_scenes'] = args.num_scenes
    if args.scenes_batch_size is not None:
        config['scenes_batch_size'] = args.scenes_batch_size
    return config


def build_hpatches_task_config(args):
    config = dict()
    if args.eval_thresholds is not None:
        config['eval_thresholds'] = args.eval_thresholds
    if args.homography_method is not None:
        config['homography_method'] = args.homography_method
    if args.homography_threshold is not None:
        config['homography_threshold'] = args.homography_threshold
    return config


def build_hpatches_benchmark_config(args):
    return {
        **build_feature_matcher_config(args),
        "dataset": build_hpatches_dataset_config(args),
        "task": build_hpatches_task_config(args)
    }
