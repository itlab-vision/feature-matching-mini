import argparse
import re
import subprocess
import sys
import logging
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # noqa: E402

from src.algorithms import (DETECTOR_DESCRIPTOR_COMPATIBILITY, DESCRIPTOR_MATCHER_COMPATIBILITY,  # noqa: E402
                            DNN_PIPELINES)

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


def run_benchmark(script_name, detector, descriptor, matcher,
                  img1, img2, device='cpu', iterations=10):
    cmd = [
        sys.executable, '-m', script_name,
        '-det', detector,
        '-des', descriptor,
        '-mat', matcher,
        '-i1', str(img1),
        '-i2', str(img2),
        '-d', device,
        '-n', str(iterations)
    ]

    logger.info(f"Running: {detector}+{descriptor}+{matcher}")

    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        output = stdout + stderr

        if process.returncode != 0:
            logger.warning(f"Failed: {detector}+{descriptor}+{matcher}")
            return False, {}

        script_type = 'staged' if 'staged' in script_name else 'pipeline'
        results = parse_performance_log(output, script_type)
        return True, results

    except Exception as e:
        logger.warning(f"Error: {e}")
        return False, {}
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()


def get_all_combinations():
    combinations = []

    for detector in DETECTOR_DESCRIPTOR_COMPATIBILITY:
        descriptors = DETECTOR_DESCRIPTOR_COMPATIBILITY.get(detector, [])

        for descriptor in descriptors:
            matchers = DESCRIPTOR_MATCHER_COMPATIBILITY.get(descriptor, [])

            for matcher in matchers:
                combinations.append((detector, descriptor, matcher))

    return combinations


def table_benchmark(img1, img2, output_csv, device='cpu', iterations=10):
    combinations = get_all_combinations()
    logger.info(f"Found {len(combinations)} valid combinations:")
    all_results = []

    for (detector, descriptor, matcher) in combinations:
        combo_result = {
            'detector': detector,
            'descriptor': descriptor,
            'matcher': matcher,
            'device': device,
            'iterations': iterations,
        }

        is_pipeline = detector in DNN_PIPELINES
        if not is_pipeline:
            success_staged, staged_results = run_benchmark(
                'samples.performance_staged_benchmark',
                detector, descriptor, matcher,
                img1, img2, device, iterations)

            if not success_staged:
                continue

            combo_result.update(staged_results)

        success_pipeline, pipeline_results = run_benchmark(
            'samples.performance_pipeline_benchmark',
            detector, descriptor, matcher,
            img1, img2, device, iterations)

        if not success_pipeline:
            continue

        combo_result.update(pipeline_results)

        all_results.append(combo_result)

    save_results_to_csv(all_results, output_csv)


def save_results_to_csv(results, output_path):
    if not results:
        logger.warning("No results to save")
        return

    df = pd.DataFrame(results)

    columns_order = [
        'detector', 'descriptor', 'matcher', 'device', 'iterations',
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
    )
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
