import argparse
import logging
import sys
from pathlib import Path
import cv2 as cv


sys.path.append(str(Path(__file__).parent.parent))  # noqa: E402

from samples.hpatches_data_manager import HPatchesDataManager  # noqa: E402
from samples.utils import (build_hpatches_benchmark_config, build_hpatches_feature_matcher_config)  # noqa: E402
from src.matchers import OpenCVMatcher  # noqa: E402
from stitching.stitcher import Stitcher  # noqa: E402

logging.basicConfig(level=logging.INFO, format='[ %(levelname)s ] %(message)s')
logger = logging.getLogger("HPatchesStitching")


def run_stitching(cli_args):
    base_config = build_hpatches_benchmark_config(cli_args)
    fm_config = build_hpatches_feature_matcher_config(cli_args)

    det = cli_args.detector
    desc = cli_args.descriptor
    mat = cli_args.matcher

    combo_name = f"{det}+{desc}+{mat}"
    output_dir = cli_args.output_dir / f"result_stitching_{combo_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running stitching for combination: {combo_name}")
    logger.info(f"Saving results to: {output_dir.resolve()}")

    stitcher = Stitcher(detector=det, descriptor=desc, matcher=mat, logger=logger, config=fm_config)
    dm = HPatchesDataManager(logger=logger, config=base_config['dataset'])
    total_processed = 0

    try:
        while dm.has_more_data():
            current_batch = dm.load_batch()
            if not current_batch:
                break

            for scene_name, scene_data in current_batch.items():
                if cli_args.viewpoint_only and not scene_name.startswith('v_'):
                    logger.info(f"Skipping non-viewpoint scene: {scene_name}")
                    continue

                img_ref = scene_data['ref_img']
                scene_dir = output_dir / scene_name

                for i, target in scene_data['targets'].items():
                    img_tgt = target['image']
                    H_gt = target['H']

                    stitched_pred = stitcher.stitch2images(img_ref, img_tgt)
                    stitched_gt = stitcher.stitch2images(img_ref, img_tgt, H_gt)

                    pair_dir = scene_dir / str(i)
                    pair_dir.mkdir(parents=True, exist_ok=True)

                    cv.imwrite(str(pair_dir / "stitched_pred.jpg"), stitched_pred)
                    cv.imwrite(str(pair_dir / "stitched_gt.jpg"), stitched_gt)

                    total_processed += 1

        dm.reset()
        logger.info(f"Successfully processed and stitched {total_processed} pairs across scenes.")

    except Exception as e:
        logger.error(f"Error during stitching process: {e}", exc_info=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run batch-based HPatches stitching for a specific algorithm combination",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    available_devices = ['cpu', 'cuda', 'mps']
    available_matchers_modes = list(OpenCVMatcher._MODE)

    parser.add_argument('-p', '--path', type=Path, required=True,
                        help='Path to hpatches-sequences-release folder')
    parser.add_argument('-o', '--output-dir', type=Path, default=Path('.'),
                        help='Root directory for output folders')

    parser.add_argument('-det', '--detector', type=str, default='sift', help='Detector name')
    parser.add_argument('-des', '--descriptor', type=str, default='sift', help='Descriptor name')
    parser.add_argument('-mat', '--matcher', type=str, default='bf', help='Matcher name')

    parser.add_argument('-d', '--device', type=str, default=None,
                        choices=available_devices, help='Device')
    parser.add_argument('-n', '--num-scenes', type=int, default=116, help='Number of scenes to process')
    parser.add_argument('-sbs', '--scenes-batch-size', type=int, default=4,
                        help='Batch size for processing scenes')
    parser.add_argument('-mp', '--modelpath', type=Path, default=None, help='Path to models')
    parser.add_argument('-vo', '--viewpoint-only', action='store_true',
                        help='Process only viewpoint scenes starting with "v_"')

    task_group = parser.add_argument_group('Homography config')
    task_group.add_argument('-et', '--eval-thresholds', type=float, nargs='+', default=[5.0],
                            help='Pixel thresholds (1.0 3.0 5.0 10.0)')
    task_group.add_argument('-hm', '--homography-method', type=str, default='ransac',
                            choices=list(Stitcher._HOMOGRAPHY_METHODS.keys()), help='Homography estimation method')
    task_group.add_argument('-ht', '--homography-threshold', type=float, default=3.0,
                            help='RANSAC reprojection threshold')

    det_group = parser.add_argument_group('Detector config')
    det_group.add_argument('-dn', '--det-nfeatures', type=int, default=None,
                           help='Max features for detector')
    det_group.add_argument('-do', '--det-noctave', type=int, default=None, help='Octave layers')
    det_group.add_argument('-dt', '--det-threshold', type=float, default=None, help='Detection threshold')

    des_group = parser.add_argument_group('Descriptor config')
    des_group.add_argument('-dsen', '--des-nfeatures', type=int, default=None,
                           help='Max features for descriptor')
    des_group.add_argument('-dsdt', '--des-threshold', type=float, default=None,
                           help='Descriptor threshold')
    des_group.add_argument('-dss', '--des-scale', type=float, default=None, help='Scale factor')

    mat_group = parser.add_argument_group('Matcher config')
    mat_group.add_argument('-mat_m', '--matcher_mode', type=str, default='simple',
                           choices=available_matchers_modes, help='Matching mode')
    mat_group.add_argument('-mr', '--mat-ratio', type=float, default=None, help='Ratio threshold for KNN')
    mat_group.add_argument('-mc', '--mat-cross-check', action='store_true', default=None,
                           help='Enable cross-check')

    return parser.parse_args()


def main():
    args = parse_args()
    if not args.path.exists():
        logger.error(f"Dataset path does not exist: {args.path}")
        return 1

    run_stitching(args)
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
