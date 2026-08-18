import sys
import logging
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # noqa: E402

from src.image_utils import read_image  # noqa: E402
from samples.utils import build_feature_matcher_config, performance_tests_parser  # noqa: E402
from src.preprocessor import Preprocessor  # noqa: E402
from src.algorithms import DNN_ALGORITHMS  # noqa: E402
from src.detectors import Detector  # noqa: E402
from src.descriptors import Descriptor  # noqa: E402
from src.matchers import Matcher  # noqa: E402
from samples.performance_profiler import PerformanceProfiler  # noqa: E402
from src.opencv_dnn_extractors import ALIKEDOpenCV, DISKOpenCV  # noqa: E402, F401
from src.opencv_dnn_matchers import LightGlueOpenCVMatcher  # noqa: E402, F401
from src.tfeat_descriptor import TFeat  # noqa: E402, F401
from src.hardnet_descriptor import HardNet  # noqa: E402, F401
from src.xfeat import XFeat  # noqa: E402, F401
from src.super_glue import SuperGlueMatcher  # noqa: E402, F401
from src.d2net import D2Net  # noqa: E402, F401

logging.basicConfig(level=logging.INFO, format='[ %(levelname)s ] %(message)s')
logger = logging.getLogger("Staged_Performance_Test_Sample")


def _dnn_staged_perf_test(logger, profiler, descriptor, descriptor_name,
                          matcher, matcher_name, img0, img1, iterations):
    times_detectors_and_descriptors = []
    times_matchers = []

    for _ in range(iterations):
        res_extract_dict0, time_desc0 = profiler.profile_dnn_extractor(descriptor, img0)
        res_extract_dict1, time_desc1 = profiler.profile_dnn_extractor(descriptor, img1)
        keypoints0 = res_extract_dict0['keypoints']
        keypoints1 = res_extract_dict1['keypoints']
        kp_index = 1 if keypoints0.ndim == 3 else 0
        count_kp0 = keypoints0.shape[kp_index]
        count_kp1 = keypoints1.shape[kp_index]

        des0 = res_extract_dict0.get('descriptors')
        descriptor_dimension = des0.shape[-1]

        times_detectors_and_descriptors.append(time_desc0)
        times_detectors_and_descriptors.append(time_desc1)
        features0 = profiler.preprocessor.prepare_features(res_extract_dict0, from_algo=descriptor_name,
                                                           to_algo=matcher_name)
        features1 = profiler.preprocessor.prepare_features(res_extract_dict1, from_algo=descriptor_name,
                                                           to_algo=matcher_name)

        res_match_dict, time_match = profiler.profile_matching(matcher, features0, features1)
        times_matchers.append(time_match)

    min_time_detectors_and_descriptors = np.min(times_detectors_and_descriptors)
    mean_time_detectors_and_descriptors = np.mean(times_detectors_and_descriptors)
    min_time_matcher = np.min(times_matchers)
    mean_time_matcher = np.mean(times_matchers)

    logger.info(f"\nMin time feature extract: {min_time_detectors_and_descriptors:.5f}\n"
                f"Mean time feature extract: {mean_time_detectors_and_descriptors:.5f}\n"
                f"Min time match: {min_time_matcher:.5f}\n"
                f"Mean time match: {mean_time_matcher:.5f}\n"
                f"Number of key points 1: {count_kp0}\n"
                f"Number of key points 2: {count_kp1}\n"
                f"Descriptors dimension: {descriptor_dimension}\n")


def _opencv_staged_perf_test(logger, profiler, detector, detector_name, descriptor, descriptor_name,
                             matcher, matcher_name, img0, img1, iterations):
    times_detectors = []
    times_descriptors = []
    times_matchers = []

    for _ in range(iterations):
        res_detect_dict0, time_detect0 = profiler.profile_detection(detector, img0)
        count_kp0 = len(res_detect_dict0['kp'])
        times_detectors.append(time_detect0)
        kp0 = profiler.preprocessor.prepare_features(res_detect_dict0,
                                                     from_algo=detector_name, to_algo=descriptor_name)
        res_desc_dict0, time_desc0 = profiler.profile_descriptor(descriptor, img0, kp0)
        des0 = res_desc_dict0.get('des')
        descriptor_dimension = des0.shape[-1]
        times_descriptors.append(time_desc0)

        res_detect_dict1, time_detect1 = profiler.profile_detection(detector, img1)
        count_kp1 = len(res_detect_dict1['kp'])
        times_detectors.append(time_detect1)
        kp1 = profiler.preprocessor.prepare_features(res_detect_dict1,
                                                     from_algo=detector_name, to_algo=descriptor_name)
        res_desc_dict1, time_desc1 = profiler.profile_descriptor(descriptor, img1, kp1)
        times_descriptors.append(time_desc1)

        features0 = profiler.preprocessor.prepare_features(res_desc_dict0, from_algo=descriptor_name,
                                                           to_algo=matcher_name)
        features1 = profiler.preprocessor.prepare_features(res_desc_dict1, from_algo=descriptor_name,
                                                           to_algo=matcher_name)

        res_match_dict, time_match = profiler.profile_matching(matcher, features0, features1)
        times_matchers.append(time_match)

    min_time_detector = np.min(times_detectors)
    mean_time_detector = np.mean(times_detectors)
    min_time_descriptor = np.min(times_descriptors)
    mean_time_descriptor = np.mean(times_descriptors)
    min_time_matcher = np.min(times_matchers)
    mean_time_matcher = np.mean(times_matchers)

    logger.info(f"\nMin time detection: {min_time_detector:.5f}\n"
                f"Mean time detection: {mean_time_detector:.5f}\n"
                f"Min time descriptor: {min_time_descriptor:.5f}\n"
                f"Mean time descriptor: {mean_time_descriptor:.5f}\n"
                f"Min time match: {min_time_matcher:.5f}\n"
                f"Mean time match: {mean_time_matcher:.5f}\n"
                f"Number of key points 1: {count_kp0}\n"
                f"Number of key points 2: {count_kp1}\n"
                f"Descriptors dimension: {descriptor_dimension}\n")


def staged_performance_test(logger, profiler, detector, detector_name, descriptor, descriptor_name,
                            matcher, matcher_name, img0, img1, iterations):

    if detector_name in DNN_ALGORITHMS or descriptor_name in DNN_ALGORITHMS:
        _dnn_staged_perf_test(logger, profiler, descriptor, descriptor_name,
                              matcher, matcher_name, img0, img1, iterations)
    else:
        _opencv_staged_perf_test(logger, profiler, detector, detector_name, descriptor, descriptor_name,
                                 matcher, matcher_name, img0, img1, iterations)


def main():
    args = performance_tests_parser()

    try:
        if not args.image1.exists() or not args.image2.exists():
            logger.error(f"One of the images does not exist: {args.image1} or {args.image2}")
            return 1

        config = build_feature_matcher_config(args)
        logger.info(f"Config: {config}")

        detector_config = config.get('detector', {})
        descriptor_config = config.get('descriptor', {})
        matcher_config = config.get('matcher', {})
        preprocessor_config = config.get('preprocessor', {})

        img0 = read_image(args.image1)
        img1 = read_image(args.image2)

        detector = Detector.create(detector_name=args.detector, logger=logger, config=detector_config)
        descriptor = Descriptor.create(descriptor_name=args.descriptor, logger=logger,
                                       config=descriptor_config)
        matcher = Matcher.create(matcher_name=args.matcher, descriptor_name=descriptor,
                                 logger=logger, config=matcher_config)
        preprocessor = Preprocessor(config=preprocessor_config, logger=logger)

        img0 = preprocessor.prepare_image(img0, from_algo='opencv', to_algo=args.detector)
        img1 = preprocessor.prepare_image(img1, from_algo='opencv', to_algo=args.detector)

        profiler = PerformanceProfiler(preprocessor)

        iterations = args.iterations
        logger.info(f"Running staged performance test with {iterations} iterations...")

        staged_performance_test(
            logger=logger,
            profiler=profiler,
            detector=detector,
            detector_name=args.detector,
            descriptor=descriptor,
            descriptor_name=args.descriptor,
            matcher=matcher,
            matcher_name=args.matcher,
            img0=img0,
            img1=img1,
            iterations=iterations
        )

        return 0

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main() or 0)
