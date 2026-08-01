import argparse
import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # noqa: E402

from src.detectors import Detector  # noqa: E402
from src.descriptors import Descriptor  # noqa: E402
from src.matchers import Matcher, OpenCVMatcher  # noqa: E402
from src.feature_matcher import FeatureMatcherCV2  # noqa: E402

from samples.utils import build_hpatches_benchmark_config  # noqa: E402
from samples.hpatches_task import HPatchesTask  # noqa: E402
from samples.hpatches_data_manager import HPatchesDataManager  # noqa: E402

logging.basicConfig(level=logging.INFO, format='[ %(levelname)s ] %(message)s')
logger = logging.getLogger("hpatches_benchmark")


def parser():
    arg_parser = argparse.ArgumentParser(
        description="HPatches Benchmark Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    available_detectors = list(Detector._METHODS.keys())
    available_descriptors = list(Descriptor._METHODS.keys())
    available_matchers = list(Matcher._METHODS.keys())
    available_matchers_modes = list(OpenCVMatcher._MODE)
    available_tasks = list(HPatchesTask._TASKS.keys())
    available_devices = ['cpu', 'cuda', 'mps']

    arg_parser.add_argument('-det', '--detector', type=str, default='sift',
                            choices=available_detectors, help='Detector algorithm')
    arg_parser.add_argument('-des', '--descriptor', type=str, default='sift',
                            choices=available_descriptors, help='Descriptor algorithm')
    arg_parser.add_argument('-mat', '--matcher', type=str, default='bf',
                            choices=available_matchers, help='Matching algorithm')

    arg_parser.add_argument('-t', '--tasks', type=str, nargs='+', default=available_tasks,
                            choices=available_tasks, help='Tasks to evaluate (default: run all available tasks)')
    arg_parser.add_argument('-d', '--device', type=str, default=None,
                            choices=available_devices, help='The device on which the script will be run')
    arg_parser.add_argument('-mp', '--modelpath', type=Path, default=None,
                            help='Path to models')

    ds_group = arg_parser.add_argument_group('Dataset config')
    ds_group.add_argument('-p', '--path', type=Path, required=True,
                          help='Path to hpatches-release folder')
    ds_group.add_argument('-n', '--num-scenes', type=int, default=116,
                          help='Number of scenes to process (default: all)')
    ds_group.add_argument('-sbs', '--scenes-batch-size', type=int, default=4,
                          help='Batch size for processing images/scenes')

    task_group = arg_parser.add_argument_group('Task config')
    task_group.add_argument('-et', '--eval-thresholds', type=float, nargs='+', default=[5.0],
                            help='Pixel thresholds for verification (1.0 3.0 5.0 10.0)')
    task_group.add_argument('-hm', '--homography-method', type=str, default='ransac',
                            choices=['ransac', 'magsac', 'lmeds', 'rho'], help='Homography estimation method')
    task_group.add_argument('-ht', '--homography-threshold', type=float, default=3.0,
                            help='Threshold for homography estimation (inlier classification)')

    det_group = arg_parser.add_argument_group('Detector config')
    det_group.add_argument('-dn', '--det-nfeatures', type=int, default=None,
                           help='Max number of features to detect')
    det_group.add_argument('-do', '--det-noctave', type=int, default=None,
                           help='Number of octave layers')
    det_group.add_argument('-dt', '--det-threshold', type=float, default=None,
                           help='Detection threshold')

    des_group = arg_parser.add_argument_group('Descriptor config')
    des_group.add_argument('-dsen', '--des-nfeatures', type=int, default=None,
                           help='Max number of features for descriptor')
    des_group.add_argument('-dsdt', '--des-threshold', type=float, default=None,
                           help='Descriptor threshold')
    des_group.add_argument('-dss', '--des-scale', type=float, default=None,
                           help='Scale factor')

    mat_group = arg_parser.add_argument_group('Matcher config')
    mat_group.add_argument('-mat_m', '--matcher_mode', type=str, default='simple',
                           choices=available_matchers_modes, help='Matching mode')
    mat_group.add_argument('-mr', '--mat-ratio', type=float, default=None,
                           help='Ratio threshold for KNN')
    mat_group.add_argument('-mc', '--mat-cross-check', action='store_true', default=None,
                           help='Enable cross-check for BF matcher')
    return arg_parser.parse_args()


def main():
    args = parser()

    try:
        if not args.path.exists():
            logger.error(f"Dataset path does not exist: {args.path}")
            return 1

        logger.info(f"Starting HPatches Benchmark. Tasks: {args.tasks}, Thresholds: {args.eval_thresholds}")

        config = build_hpatches_benchmark_config(args)
        feature_matcher = FeatureMatcherCV2(detector=args.detector, descriptor=args.descriptor,
                                            matcher=args.matcher, logger=logger, config=config)
        dm = HPatchesDataManager(logger=logger, config=config['dataset'])

        task_objects = {}
        results_by_task = {}
        for task_name in args.tasks:
            task_objects[task_name] = HPatchesTask.create(task_name=task_name, logger=logger,
                                                          config=config['task'])
            results_by_task[task_name] = {}

        while dm.has_more_data():
            current_batch = dm.load_batch()
            if not current_batch:
                break

            for scene_name, data in current_batch.items():
                scene_matching_data = {scene_name: {}}
                img_ref = data['ref_img']

                for i, target in data['targets'].items():
                    img_tgt = target['image']
                    H = target['H']

                    features_ref, features_tgt, correspondences = feature_matcher.match(img_ref, img_tgt)
                    scene_matching_data[scene_name][i] = {
                        'kp_ref': features_ref['kp'],
                        'kp_tgt': features_tgt['kp'],
                        'matches': correspondences['matches'],
                        'H': H,
                        'ref_shape': data['ref_shape'],
                        'tgt_shape': target['tgt_shape'],
                    }

                for task_name, task_obj in task_objects.items():
                    results_scene = task_obj.eval_task(scene_matching_data, [scene_name])
                    if task_name not in results_by_task:
                        results_by_task[task_name] = results_scene
                    else:
                        for threshold in results_scene:
                            if threshold not in results_by_task[task_name]:
                                results_by_task[task_name][threshold] = {}
                            results_by_task[task_name][threshold].update(results_scene[threshold])

        for task_name, task_obj in task_objects.items():
            task_obj.report_metrics(results_by_task[task_name], f"End-to-End Pipeline [{task_name}]")

        logger.info("Pipeline finished successfully")
        return 0

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main() or 0)
