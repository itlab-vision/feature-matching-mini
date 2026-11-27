import argparse
import re
import subprocess
import sys
import logging
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # noqa: E402

from src.algorithms import (DETECTOR_DESCRIPTOR_COMPATIBILITY, DESCRIPTOR_MATCHER_COMPATIBILITY,  # noqa: E402
                            TORCH_PIPELINES, get_compatible_descriptors, get_compatible_matchers)

logging.basicConfig(level=logging.INFO, format='[ %(levelname)s ] %(message)s')
logger = logging.getLogger("PerformanceBenchmark")


def parse_performance_log(log_output, script_type):
    results = {}

    if script_type == 'staged':
        patterns = {
            'min_detection_ms': r'Min time detection:\s*([\d\.eE\+\-]+)',
            'mean_detection_ms': r'Mean time detection:\s*([\d\.eE\+\-]+)',
            'min_descriptor_ms': r'Min time descriptor:\s*([\d\.eE\+\-]+)',
            'mean_descriptor_ms': r'Mean time descriptor:\s*([\d\.eE\+\-]+)',
            'min_extractor_ms': r'Min time feature extract:\s*([\d\.eE\+\-]+)',
            'mean_extractor_ms': r'Mean time feature extract:\s*([\d\.eE\+\-]+)',
            'min_match_ms': r'Min time match:\s*([\d\.eE\+\-]+)',
            'mean_match_ms': r'Mean time match:\s*([\d\.eE\+\-]+)',
            'num_keypoints_1': r'Number of key points 1:\s*(\d+)',
            'num_keypoints_2': r'Number of key points 2:\s*(\d+)',
            'descriptors_dim': r'Descriptors dimension:\s*(\d+)'
        }
    else:
        patterns = {
            'min_pipeline_ms': r'Min time pipeline test:\s*([\d\.eE\+\-]+)',
            'mean_pipeline_ms': r'Mean time pipeline test:\s*([\d\.eE\+\-]+)',
            'num_keypoints_1': r'Number of key points 1:\s*(\d+)',
            'num_keypoints_2': r'Number of key points 2:\s*(\d+)'
        }

    for key, pattern in patterns.items():
        match = re.search(pattern, log_output)
        if match:
            value = float(match.group(1))
            if key in ['num_keypoints_1', 'num_keypoints_2', 'descriptors_dim']:
                results[key] = int(value)
            else:
                results[key] = value * 1000

    return results


def get_model_args(combo, model_dir, num_keypoints):
    detector = combo['detector']
    detector_backend = combo['detector_backend']
    matcher_backend = combo['matcher_backend']
    features = detector.replace('_lightglue', '')
    args = ['-dn', str(num_keypoints), '-dsen', str(num_keypoints)]
    if detector in ('sift_lightglue', 'doghardnet_lightglue'):
        args += ['-dt', '0', '-dsdt', '0']

    if detector_backend in ('onnx', 'opencv'):
        if detector == 'disk_lightglue':
            args += ['-mpd', str(model_dir / 'disk_1024.onnx')]
        elif detector == 'aliked_lightglue':
            args += ['-mpa', str(model_dir / 'aliked_lightglue.onnx')]
        elif detector == 'superpoint_lightglue':
            args += ['--superpoint-model-path', str(model_dir / 'superpoint_lightglue.onnx')]
    elif detector_backend == 'executorch':
        input_shapes = {
            'disk_lightglue': (1, 3, 1024, 1024),
            'superpoint_lightglue': (1, 3, 480, 640),
            'aliked_lightglue': (1, 3, 640, 640),
        }
        args += ['--det-et-model', str(model_dir / f'{detector}_xnnpack.pte'),
                 '--det-et-input-shape', *map(str, input_shapes[detector]),
                 '--det-et-num-keypoints', str(num_keypoints)]

    if matcher_backend in ('onnx', 'opencv'):
        if matcher_backend == 'opencv':
            model_name = f'lightglue_{features}_opencv.onnx'
        else:
            model_name = f'lightglue_{features}_k{num_keypoints}.onnx'
        args.append('--mat-normalize-keypoints')
        args += ['-mplg', str(model_dir / model_name)]
    elif matcher_backend == 'executorch':
        args += ['--mat-et-model', str(model_dir / f'lightglue_{features}_k{num_keypoints}_xnnpack.pte'),
                 '--mat-et-num-keypoints', str(num_keypoints), '--mat-normalize-keypoints']
    return args


def run_benchmark(script_name, combo, img1, img2, device='cpu', iterations=10,
                  model_dir=Path('results_6_11/models'), num_keypoints=512, log_dir=None):
    det = combo['detector']
    detb = combo['detector_backend']
    des = combo['descriptor']
    desb = combo['descriptor_backend']
    mat = combo['matcher']
    matb = combo['matcher_backend']
    cmd = [
        sys.executable, '-m', script_name,
        '-det', det,
        '-detb', detb,
        '-des', des,
        '-desb', desb,
        '-mat', mat,
        '-matb', matb,
        '-i1', str(img1),
        '-i2', str(img2),
        '-d', device,
        '-n', str(iterations),
        *get_model_args(combo, model_dir, num_keypoints),
    ]

    logger.info(f"Running: {det} ({detb}) + {des} ({desb}) + {mat} ({matb})")
    process = None
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        output = stdout + stderr
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f'{det}_{detb}_{matb}_{script_name.rsplit(".", 1)[-1]}.log'
            log_path.write_text(f"Command: {' '.join(cmd)}\n{output}", encoding='utf-8')

        if process.returncode != 0 or 'inference error:' in output:
            logger.warning(f"Failed: {det} ({detb}) + {des} ({desb}) + {mat} ({matb})")
            return False, {}
        script_type = 'staged' if 'staged' in script_name else 'pipeline'
        return True, parse_performance_log(output, script_type)
    except Exception as e:
        logger.warning(f"Error: {e}")
        return False, {}
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()


def get_all_combinations(methods=None, backends=None):
    combinations = []
    methods = set(methods) if methods else None
    backends = set(backends) if backends else None
    mixed_extractors = {'sift_lightglue', 'doghardnet_lightglue'}
    for detector, det_backends_dict in DETECTOR_DESCRIPTOR_COMPATIBILITY.items():
        if methods is not None and detector not in methods:
            continue
        for det_backend, desc_backends_dict in det_backends_dict.items():
            for desc_backend in desc_backends_dict.keys():
                descriptors = get_compatible_descriptors(detector, det_backend, desc_backend)
                for descriptor in descriptors:
                    match_backends_dict = DESCRIPTOR_MATCHER_COMPATIBILITY.get(descriptor, {})
                    if desc_backend in match_backends_dict:
                        for match_backend in match_backends_dict[desc_backend].keys():
                            if backends is not None:
                                if detector in mixed_extractors:
                                    if det_backend != 'torch' or desc_backend != 'torch' or \
                                            match_backend not in backends:
                                        continue
                                elif not (det_backend == desc_backend == match_backend and match_backend in backends):
                                    continue
                            matchers = get_compatible_matchers(descriptor, desc_backend, match_backend)
                            for matcher in matchers:
                                combo = {
                                    'detector': detector,
                                    'detector_backend': det_backend,
                                    'descriptor': descriptor,
                                    'descriptor_backend': desc_backend,
                                    'matcher': matcher,
                                    'matcher_backend': match_backend
                                }
                                combinations.append(combo)

    return combinations


def table_benchmark(img1, img2, output_csv, device='cpu', iterations=10, methods=None, backends=None,
                    model_dir=Path('results_6_11/models'), num_keypoints=512):
    combinations = get_all_combinations(methods, backends)
    logger.info(f"Found {len(combinations)} valid combinations:")
    all_results = []
    log_dir = output_csv.parent / 'logs' / 'runs'

    for combo in combinations:
        combo_result = {
            'detector': combo['detector'],
            'detector_backend': combo['detector_backend'],
            'descriptor': combo['descriptor'],
            'descriptor_backend': combo['descriptor_backend'],
            'matcher': combo['matcher'],
            'matcher_backend': combo['matcher_backend'],
            'device': device,
            'iterations': iterations,
        }

        is_pipeline = combo['detector'] in TORCH_PIPELINES
        if not is_pipeline:
            success_staged, staged_results = run_benchmark(
                'samples.performance_staged_benchmark', combo, img1, img2, device, iterations,
                model_dir, num_keypoints, log_dir)

            if not success_staged:
                continue

            combo_result.update(staged_results)

        success_pipeline, pipeline_results = run_benchmark(
            'samples.performance_pipeline_benchmark', combo, img1, img2, device, iterations,
            model_dir, num_keypoints, log_dir)

        if not success_pipeline:
            continue

        combo_result.update(pipeline_results)
        all_results.append(combo_result)
        save_results_to_csv(all_results, output_csv)

    save_results_to_csv(all_results, output_csv)


def save_results_to_csv(results, output_path):
    if not results:
        logger.warning("No results to save")
        return

    df = pd.DataFrame(results)

    columns_order = [
        'detector', 'detector_backend',
        'descriptor', 'descriptor_backend',
        'matcher', 'matcher_backend',
        'device', 'iterations',
        'min_detection_ms', 'mean_detection_ms',
        'min_descriptor_ms', 'mean_descriptor_ms',
        "min_extractor_ms", "mean_extractor_ms",
        'num_keypoints_1', 'num_keypoints_2', 'descriptors_dim',
        'min_match_ms', 'mean_match_ms',
        'min_pipeline_ms', 'mean_pipeline_ms',
    ]

    existing_columns = [col for col in columns_order if col in df.columns]
    df = df[existing_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format='%.2f')
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Total rows: {len(df)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run performance benchmarks with all combinations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    available_devices = ['cpu', 'cuda', 'mps']

    parser.add_argument('-i1', '--image1', type=Path, required=True,
                        help='Path to the first image')
    parser.add_argument('-i2', '--image2', type=Path, required=True,
                        help='Path to the second image')

    parser.add_argument('-o', '--output', type=Path, default=Path('benchmark_results.csv'),
                        help='Output CSV file path')

    parser.add_argument('-d', '--device', type=str, default='cpu',
                        choices=available_devices, help='The device on which the script will be run')

    parser.add_argument('-n', '--iterations', type=int, default=10,
                        help='Number of iterations for performance testing (default: 10)')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Detector/descriptor names to benchmark')
    parser.add_argument('--backends', nargs='+', default=None,
                        choices=['torch', 'onnx', 'opencv', 'executorch'],
                        help='Backends to benchmark')
    parser.add_argument('--model-dir', type=Path, default=Path('results_6_11/models'),
                        help='Directory with converted models')
    parser.add_argument('--num-keypoints', type=int, default=512,
                        help='Number of features used by all pipelines')

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.image1.exists() or not args.image2.exists():
        logger.error(f"One of the images does not exist: {args.image1} or {args.image2}")
        return 1

    table_benchmark(
        img1=args.image1,
        img2=args.image2,
        output_csv=args.output,
        device=args.device,
        iterations=args.iterations,
        methods=args.methods,
        backends=args.backends,
        model_dir=args.model_dir,
        num_keypoints=args.num_keypoints,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
