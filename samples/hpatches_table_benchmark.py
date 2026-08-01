import argparse
import copy
import sys
import numpy as np
import logging
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # noqa: E402

from src.matchers import OpenCVMatcher  # noqa: E402
from src.algorithms import DETECTOR_DESCRIPTOR_COMPATIBILITY, DESCRIPTOR_MATCHER_COMPATIBILITY  # noqa: E402
from src.feature_matcher import FeatureMatcherCV2  # noqa: E402

from samples.utils import build_hpatches_benchmark_config, build_feature_matcher_config  # noqa: E402
from samples.hpatches_task import HPatchesTask  # noqa: E402
from samples.hpatches_data_manager import HPatchesDataManager  # noqa: E402


logging.basicConfig(level=logging.INFO, format='[ %(levelname)s ] %(message)s')
logger = logging.getLogger("HPatchesTableBenchmark")


def get_all_combinations():
    combinations = []
    for detector in DETECTOR_DESCRIPTOR_COMPATIBILITY:
        descriptors = DETECTOR_DESCRIPTOR_COMPATIBILITY.get(detector, [])
        for descriptor in descriptors:
            matchers = DESCRIPTOR_MATCHER_COMPATIBILITY.get(descriptor, [])
            for matcher in matchers:
                combinations.append((detector, descriptor, matcher))
    return combinations


def load_existing_results(output_path):
    if not output_path.exists():
        logger.info(f"No existing results found at {output_path}")
        return None, set()

    try:
        df = pd.read_csv(output_path)
        logger.info(f"Loaded {len(df)} existing results from {output_path}")

        existing_combos = set()
        for _, row in df.iterrows():
            combo = (row['detector'], row['descriptor'], row['matcher'])
            existing_combos.add(combo)

        logger.info(f"Found {len(existing_combos)} unique combinations already computed")
        return df, existing_combos

    except Exception as e:
        logger.warning(f"Error loading existing results: {e}")
        return None, set()


def save_single_result(output_path, new_result, tasks, thresholds):
    base_columns = ['detector', 'descriptor', 'matcher', 'device', 'num_scenes']
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

    cli_args.detector = all_combinations[0][0]
    cli_args.descriptor = all_combinations[0][1]
    cli_args.matcher = all_combinations[0][2]

    base_config = build_hpatches_benchmark_config(cli_args)
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
    feature_matcher = FeatureMatcherCV2(detector=combinations[0][0], descriptor=combinations[0][1],
                                        matcher=combinations[0][2], logger=logger, config=base_config)
    dm = HPatchesDataManager(logger=logger, config=base_config['dataset'])

    for idx, combo in enumerate(combinations, 1):
        det, desc, mat = combo
        logger.info(f"{idx}/{len(combinations)} Processing: {det} + {desc} + {mat}")

        current_combo_results = {task: {} for task in cli_args.tasks}

        tmp_args = copy.copy(cli_args)
        tmp_args.detector, tmp_args.descriptor, tmp_args.matcher = det, desc, mat
        combo_fm_config = build_feature_matcher_config(tmp_args)
        feature_matcher.__init__(detector=det, descriptor=desc, matcher=mat,
                                 logger=logger, config=combo_fm_config)

        try:
            while dm.has_more_data():
                current_batch = dm.load_batch()
                if not current_batch:
                    break

                for scene_name, scene_data in current_batch.items():
                    img_ref = scene_data['ref_img']
                    scene_matching_data = {scene_name: {}}

                    for i, target in scene_data['targets'].items():
                        features_ref, features_tgt, correspondences = feature_matcher.match(img_ref, target['image'])
                        scene_matching_data[scene_name][i] = {
                            'kp_ref': features_ref['kp'], 'kp_tgt': features_tgt['kp'],
                            'matches': correspondences['matches'], 'H': target['H'],
                            'ref_shape': scene_data['ref_shape'], 'tgt_shape': target['tgt_shape'],
                        }

                    for name, task_obj in task_objects.items():
                        res_scene = task_obj.eval_task(scene_matching_data, [scene_name])

                        for threshold in res_scene:
                            if threshold not in current_combo_results[name]:
                                current_combo_results[name][threshold] = {}
                            current_combo_results[name][threshold].update(res_scene[threshold])

            dm.reset()
            combo_row = {
                'detector': det, 'descriptor': desc, 'matcher': mat,
                'device': cli_args.device or 'cpu',
                'num_scenes': cli_args.num_scenes,
            }

            for name, task_obj in task_objects.items():
                per_threshold_metrics = task_obj.report_metrics(current_combo_results[name])
                combo_row.update(flatten_metrics(name, per_threshold_metrics))

            save_single_result(cli_args.output, combo_row, cli_args.tasks, cli_args.eval_thresholds)
            logger.info(f"Combination {det}+{desc}+{mat} finished and saved")

        except Exception as e:
            logger.error(f"Failed combination {det}+{desc}+{mat}: {e}")
            continue


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HPatches benchmarks for all combinations with incremental saving",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    available_devices = ['cpu', 'cuda', 'mps']
    available_tasks = list(HPatchesTask._TASKS.keys())
    available_matchers_modes = list(OpenCVMatcher._MODE)

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

    des_group = parser.add_argument_group('Descriptor config')
    des_group.add_argument('-dsen', '--des-nfeatures', type=int, default=None,
                           help='Max number of features for descriptor')
    des_group.add_argument('-dsdt', '--des-threshold', type=float, default=None,
                           help='Descriptor threshold')
    des_group.add_argument('-dss', '--des-scale', type=float, default=None,
                           help='Scale factor')

    mat_group = parser.add_argument_group('Matcher config')
    mat_group.add_argument('-mat_m', '--matcher_mode', type=str, default='simple',
                           choices=available_matchers_modes, help='Matching mode')
    mat_group.add_argument('-mr', '--mat-ratio', type=float, default=None,
                           help='Ratio threshold for KNN')
    mat_group.add_argument('-mc', '--mat-cross-check', action='store_true', default=None,
                           help='Enable cross-check for BF matcher')

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
