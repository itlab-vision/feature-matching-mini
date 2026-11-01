import argparse
import copy
import sys
from itertools import product
import numpy as np
import logging
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # noqa: E402

from src.algorithms import (BACKEND_ALGORITHMS, OPENCV_MATCHERS_MODE,  # noqa: E402
                            get_compatible_descriptors, get_compatible_matchers)
from src.feature_matcher import FeatureMatcherCV2  # noqa: E402

from src.backend.executorch.model_wrapper.model_wrapper import (  # noqa: E402
    EXECUTORCH_DESCRIPTOR_MODELS,
    EXECUTORCH_DETECTOR_MODELS,
    EXECUTORCH_MATCHER_MODELS,
)
from src.backend.opencv.model_loader.model_loader import (  # noqa: E402
    OPENCV_DESCRIPTOR_MODELS, OPENCV_DETECTOR_MODELS, OPENCV_MATCHER_MODELS,
)
import src.backend.opencv.inference_api  # noqa: E402, F401
import src.backend.opencv.io_adapter  # noqa: E402, F401
import src.backend.onnx.model_loader  # noqa: E402, F401
import src.backend.onnx.inference_api  # noqa: E402, F401
import src.backend.onnx.io_adapter  # noqa: E402, F401
from samples.utils import build_hpatches_benchmark_config, build_hpatches_feature_matcher_config  # noqa: E402
from samples.hpatches_task import HPatchesTask  # noqa: E402
from samples.hpatches_data_manager import HPatchesDataManager  # noqa: E402


logging.basicConfig(level=logging.INFO, format='[ %(levelname)s ] %(message)s')
logger = logging.getLogger("HPatchesTableBenchmark")


def get_all_combinations():
    combinations = []
    backends = BACKEND_ALGORITHMS.keys()

    for detector_backend, descriptor_backend, matcher_backend in product(backends, repeat=3):
        detectors = BACKEND_ALGORITHMS[detector_backend][0]
        for detector in sorted(detectors):
            descriptors = get_compatible_descriptors(detector, detector_backend, descriptor_backend)
            for descriptor in descriptors:
                matchers = get_compatible_matchers(descriptor, descriptor_backend, matcher_backend)
                for matcher in matchers:
                    combinations.append((detector, detector_backend, descriptor, descriptor_backend,
                                         matcher, matcher_backend))
    return combinations


def configure_model_paths(args):
    model_path = args.modelpath or Path('models')

    if args.det_backend == 'executorch':
        args.det_et_model = model_path / EXECUTORCH_DETECTOR_MODELS[args.detector]
    elif args.det_backend == 'opencv' and args.detector in OPENCV_DETECTOR_MODELS:
        setattr(args, f'{args.detector}_model_path', model_path / OPENCV_DETECTOR_MODELS[args.detector])

    if args.des_backend == 'executorch':
        args.des_et_model = model_path / EXECUTORCH_DESCRIPTOR_MODELS[args.descriptor]
    elif args.des_backend == 'opencv' and args.descriptor in OPENCV_DESCRIPTOR_MODELS:
        setattr(args, f'{args.descriptor}_model_path', model_path / OPENCV_DESCRIPTOR_MODELS[args.descriptor])

    if args.mat_backend == 'executorch':
        args.mat_et_model = model_path / EXECUTORCH_MATCHER_MODELS[args.descriptor]
        args.mat_et_num_keypoints = 256
    elif args.mat_backend == 'opencv' and args.matcher == 'lightglue':
        args.lightglue_model_path = model_path / OPENCV_MATCHER_MODELS[args.descriptor]


def load_existing_results(output_path):
    if not output_path.exists():
        logger.info(f"No existing results found at {output_path}")
        return None, set()

    try:
        df = pd.read_csv(output_path)
        logger.info(f"Loaded {len(df)} existing results from {output_path}")

        existing_combos = set()
        backend_columns = {'detector_backend', 'descriptor_backend', 'matcher_backend'}
        if backend_columns.issubset(df.columns):
            for _, row in df.iterrows():
                combo = (
                    row['detector'], row['detector_backend'],
                    row['descriptor'], row['descriptor_backend'],
                    row['matcher'], row['matcher_backend'],
                )
                existing_combos.add(combo)
        elif len(df) > 0:
            logger.warning("Existing results do not contain backend columns and will not be skipped")

        logger.info(f"Found {len(existing_combos)} unique combinations already computed")
        return df, existing_combos

    except Exception as e:
        logger.warning(f"Error loading existing results: {e}")
        return None, set()


def save_single_result(output_path, new_result, tasks, thresholds):
    base_columns = [
        'detector', 'detector_backend',
        'descriptor', 'descriptor_backend',
        'matcher', 'matcher_backend',
        'device', 'num_scenes',
    ]
    metric_columns = []
    for task in tasks:
        t = task.lower()
        for threshold in thresholds:
            if t == 'matchingap':
                metric_columns.append(f'matchingap_mean_ap_{threshold}')
            elif t == 'matchingscore':
                metric_columns.extend([f'matchingscore_mean_ms_{threshold}', f'matchingscore_mean_prec_{threshold}'])
            elif t == 'homographyauc':
                metric_columns.append(f'homographyauc_mean_auc_{threshold}')

    columns_order = base_columns + metric_columns

    if output_path.exists():
        df = pd.read_csv(output_path)
    else:
        df = pd.DataFrame(columns=columns_order)

    new_df = pd.DataFrame([new_result])
    df = pd.concat([df, new_df], ignore_index=True)

    existing_cols = [col for col in columns_order if col in df.columns]
    df = df[existing_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def flatten_metrics(task_name, per_threshold):
    flat = {}
    if per_threshold is None:
        return flat

    for threshold, m in per_threshold.items():
        if m is None:
            logger.warning(f"No {task_name} results for threshold {threshold}px")
            continue
        for metric_key, value in m.items():
            if isinstance(value, (int, float, np.float32, np.float64)):
                flat[f"{task_name.lower()}_{metric_key}_{threshold}"] = round(float(value), 5)
            else:
                flat[f"{task_name.lower()}_{metric_key}_{threshold}"] = value
    return flat


def table_benchmark(cli_args):
    all_combinations = get_all_combinations()
    if not all_combinations:
        logger.error("No combinations found in algorithms.py")
        return

    first_combination = all_combinations[0]
    base_args = copy.copy(cli_args)
    base_args.detector, base_args.det_backend = first_combination[:2]
    base_args.descriptor, base_args.des_backend = first_combination[2:4]
    base_args.matcher, base_args.mat_backend = first_combination[4:]
    configure_model_paths(base_args)

    base_config = build_hpatches_benchmark_config(base_args)
    _, existing_combos = load_existing_results(cli_args.output)

    logger.info(f"Found {len(all_combinations)} valid combinations")
    logger.info(f"Tasks to run: {cli_args.tasks}")
    logger.info(f"Thresholds: {cli_args.eval_thresholds}")

    if not cli_args.no_skip:
        combinations = [c for c in all_combinations if c not in existing_combos]
        logger.info(f"Remaining: {len(combinations)} combinations")
    else:
        combinations = all_combinations

    if not combinations:
        logger.info("All combinations already computed")
        return

    task_objects = {name: HPatchesTask.create(task_name=name, logger=logger, config=base_config['task'])
                    for name in cli_args.tasks}
    dm = HPatchesDataManager(logger=logger, config=base_config['dataset'])

    for idx, combo in enumerate(combinations, 1):
        det, det_backend, desc, des_backend, mat, mat_backend = combo
        logger.info(f"{idx}/{len(combinations)} Processing: "
                    f"{det}[{det_backend}] + {desc}[{des_backend}] + {mat}[{mat_backend}]")

        current_combo_results = {task: {} for task in cli_args.tasks}

        tmp_args = copy.copy(cli_args)
        tmp_args.detector, tmp_args.descriptor, tmp_args.matcher = det, desc, mat
        tmp_args.det_backend = det_backend
        tmp_args.des_backend = des_backend
        tmp_args.mat_backend = mat_backend
        configure_model_paths(tmp_args)
        combo_fm_config = build_hpatches_feature_matcher_config(tmp_args)
        feature_matcher = None

        try:
            feature_matcher = FeatureMatcherCV2(detector=det, descriptor=desc, matcher=mat,
                                                logger=logger, config=combo_fm_config)

            while dm.has_more_data():
                current_batch = dm.load_batch()
                if not current_batch:
                    break

                for scene_name, scene_data in current_batch.items():
                    img_ref = scene_data['ref_img']
                    scene_matching_data = {scene_name: {}}

                    for i, target in scene_data['targets'].items():
                        features_ref, features_tgt, correspondences = feature_matcher.match(img_ref, target['image'])
                        if 'keypoints0' in correspondences and 'keypoints1' in correspondences:
                            keypoints0 = correspondences['keypoints0']
                            keypoints1 = correspondences['keypoints1']
                        elif 'keypoints' in features_ref and 'keypoints' in features_tgt:
                            keypoints0 = features_ref['keypoints']
                            keypoints1 = features_tgt['keypoints']
                        else:
                            logger.error(f"Could not find keypoints for scene {scene_name}")
                            raise

                        scene_matching_data[scene_name][i] = {
                            'kp_ref': keypoints0, 'kp_tgt': keypoints1,
                            'matches': correspondences['matches'], 'H': target['H'],
                            'ref_shape': scene_data['ref_shape'], 'tgt_shape': target['tgt_shape'],
                        }

                    for name, task_obj in task_objects.items():
                        res_scene = task_obj.eval_task(scene_matching_data, [scene_name])

                        for threshold in res_scene:
                            if threshold not in current_combo_results[name]:
                                current_combo_results[name][threshold] = {}
                            current_combo_results[name][threshold].update(res_scene[threshold])

            combo_row = {
                'detector': det, 'detector_backend': det_backend,
                'descriptor': desc, 'descriptor_backend': des_backend,
                'matcher': mat, 'matcher_backend': mat_backend,
                'device': cli_args.device or 'cpu',
                'num_scenes': cli_args.num_scenes,
            }

            for name, task_obj in task_objects.items():
                per_threshold_metrics = task_obj.report_metrics(current_combo_results[name])
                combo_row.update(flatten_metrics(name, per_threshold_metrics))

            save_single_result(cli_args.output, combo_row, cli_args.tasks, cli_args.eval_thresholds)
            logger.info(f"Combination {det}[{det_backend}]+{desc}[{des_backend}]+"
                        f"{mat}[{mat_backend}] finished and saved")

        except Exception as e:
            logger.error(f"Failed combination {det}[{det_backend}]+{desc}[{des_backend}]+"
                         f"{mat}[{mat_backend}]: {e}")
            continue
        finally:
            dm.reset()
            if feature_matcher is not None:
                feature_matcher.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HPatches benchmarks for all combinations with incremental saving",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    available_devices = ['cpu', 'cuda', 'mps']
    available_tasks = list(HPatchesTask._TASKS.keys())

    parser.add_argument('-p', '--path', type=Path, required=True,
                        help='Path to hpatches-sequences-release folder')
    parser.add_argument('-o', '--output', type=Path, default=Path('hpatches_results.csv'),
                        help='Output CSV file path')
    parser.add_argument('-t', '--tasks', type=str, nargs='+', choices=available_tasks,
                        default=available_tasks, help='Tasks to run')

    parser.add_argument('-d', '--device', type=str, default=None, choices=available_devices,
                        help='Device to run on')
    parser.add_argument('-n', '--num-scenes', type=int, default=116,
                        help='Number of scenes to process')
    parser.add_argument('-mp', '--modelpath', type=Path, default=None,
                        help='Path to models')
    parser.add_argument('-sbs', '--scenes-batch-size', type=int, default=4,
                        help='Batch size for processing images/scenes')
    parser.add_argument('--no-skip', action='store_true',
                        help='Recompute already existing combinations')

    task_group = parser.add_argument_group('Task config')
    task_group.add_argument('-et', '--eval-thresholds', type=float, nargs='+', default=[5.0],
                            help='Pixel thresholds (1.0 3.0 5.0 10.0)')
    task_group.add_argument('-hm', '--homography-method', type=str, default='ransac',
                            choices=['ransac', 'magsac', 'lmeds', 'rho'], help='Homography estimation method')
    task_group.add_argument('-ht', '--homography-threshold', type=float, default=3.0,
                            help='Threshold for homography estimation')

    det_group = parser.add_argument_group('Detector config')
    det_group.add_argument('-dn', '--det-nfeatures', type=int, default=None,
                           help='Max number of features to detect')
    det_group.add_argument('-do', '--det-noctave', type=int, default=None,
                           help='Number of octave layers')
    det_group.add_argument('-dt', '--det-threshold', type=float, default=None,
                           help='Detection threshold')
    det_group.add_argument('--det-et-model', type=Path,
                           help='Path to a detector .pte file')
    det_group.add_argument('--det-et-input-shape', type=int, nargs=4, metavar=('N', 'C', 'H', 'W'),
                           help='Static detector .pte input shape')
    det_group.add_argument('--det-et-num-keypoints', type=int, default=None,
                           help='Number of keypoints emitted by the ExecuTorch detector')

    des_group = parser.add_argument_group('Descriptor config')
    des_group.add_argument('-dsen', '--des-nfeatures', type=int, default=None,
                           help='Max number of features for descriptor')
    des_group.add_argument('-dsdt', '--des-threshold', type=float, default=None,
                           help='Descriptor threshold')
    des_group.add_argument('-dss', '--des-scale', type=float, default=None,
                           help='Scale factor')
    des_group.add_argument('--des-et-model', type=Path,
                           help='Path to a descriptor .pte file')
    des_group.add_argument('--des-et-patch-size', type=int, default=32,
                           help='Patch size for descriptor model')

    mat_group = parser.add_argument_group('Matcher config')
    mat_group.add_argument('-mat_m', '--matcher_mode', type=str, default='simple',
                           choices=OPENCV_MATCHERS_MODE, help='Matching mode')
    mat_group.add_argument('-mr', '--mat-ratio', type=float, default=None,
                           help='Ratio threshold for KNN')
    mat_group.add_argument('-mc', '--mat-cross-check', action='store_true', default=None,
                           help='Enable cross-check for BF matcher')
    mat_group.add_argument('--mat-et-model', type=Path,
                           help='Path to a fixed-K matcher .pte file')
    mat_group.add_argument('--mat-et-num-keypoints', type=int, default=None,
                           help='Number of keypoints expected by the ExecuTorch matcher')

    return parser.parse_args()


def main():
    args = parse_args()
    if not args.path.exists():
        logger.error(f"Dataset path does not exist: {args.path}")
        return 1

    logger.info("HPatches Benchmark Table Generator")
    table_benchmark(args)
    logger.info("Benchmark completed successfully")

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
