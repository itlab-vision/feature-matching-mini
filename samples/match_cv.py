import argparse
import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # noqa: E402

from src.image_utils import read_image, save_image, show_image  # noqa: E402
from samples.utils import build_feature_matcher_config  # noqa: E402
from src.detectors import Detector  # noqa: E402
from src.descriptors import Descriptor  # noqa: E402
from src.matchers import Matcher, OpenCVMatcher  # noqa: E402
from src.feature_matcher import FeatureMatcherCV2  # noqa: E402


logging.basicConfig(level=logging.INFO, format='[ %(levelname)s ] %(message)s')
logger = logging.getLogger("CV_Sample")


def parser():
    arg_parser = argparse.ArgumentParser(
        description="Matching points in two images using OpenCV algorithms",
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
    arg_parser.add_argument('-s', '--save', type=Path, default=None,
                            help='Path to save result image')

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
    det_group.add_argument('--det-et-model', type=Path,
                            help='Path to a detector .pte file')
    det_group.add_argument('--det-et-input-shape', type=int, nargs=4, metavar=('N', 'C', 'H', 'W'),
                           help='Static detector .pte input shape')
    det_group.add_argument('--det-et-num-keypoints', type=int, default=256,
                           help='Fixed K emitted by the ExecuTorch detector')

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
    des_group.add_argument('--des-et-model', type=Path,
                           help='Path to a descriptor .pte file')
    des_group.add_argument('--des-et-patch-size', type=int, default=32,
                           help='Patch size for descriptor model')

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
    mat_group.add_argument('--mat-et-model', type=Path,
                           help='Path to a fixed-K matcher .pte file')
    return arg_parser.parse_args()


def main():
    args = parser()

    try:
        if not args.image1.exists() or not args.image2.exists():
            logger.error(f"One of the images does not exist: {args.image1} or {args.image2}")
            return 1

        logger.info("Starting Feature Matching Pipeline")
        logger.info(f"Comparing pair: {args.image1} and {args.image2}")

        config = build_feature_matcher_config(args)
        logger.info(f"Config: {config}")

        img1 = read_image(args.image1)
        img2 = read_image(args.image2)

        feature_matcher = FeatureMatcherCV2(detector=args.detector, descriptor=args.descriptor,
                                            matcher=args.matcher, logger=logger, config=config)
        features0, features1, correspondences = feature_matcher.match(img1, img2)
        res_img = feature_matcher.visualize_matches(img1, features0, img2, features1, correspondences)

        if args.save:
            save_image(res_img, save_path=args.save)
            logger.info(f"Result successfully saved to: {args.save}")

        show_image(res_img, title="Matched Image")
        logger.info("Pipeline finished successfully")
        return 0

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main() or 0)
