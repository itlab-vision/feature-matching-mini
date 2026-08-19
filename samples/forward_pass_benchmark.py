import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # noqa: E402

from samples.forward_profiler import ForwardProfiler  # noqa: E402

logging.basicConfig(level=logging.INFO, format='[ %(levelname)s ] %(message)s')
logger = logging.getLogger("ForwardPassBenchmark")


def parser():
    arg_parser = argparse.ArgumentParser(
        description="Forward pass analysis for OpenCV DNN, PyTorch and ExecuTorch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument("-p", "--platform", required=True,
                            choices=["opencv", "pytorch", "executorch"], help="Target inference platform")
    arg_parser.add_argument("-m", "--modelpath", required=True, type=str,
                            help="Path to model file (.onnx/.pte)")
    arg_parser.add_argument("-sh", "--shape", nargs="+", type=int,
                            default=[1, 3, 640, 640], help="Input tensor shape")
    arg_parser.add_argument("-w", "--warmup", type=int, default=5,
                            help="Number of warmup iterations before measurement")
    arg_parser.add_argument("-n", "--iterations", type=int, default=10,
                            help="Number of measured iterations")

    arg_parser.add_argument("-d", "--device", type=str, default="cpu",
                            choices=["cpu", "cuda"],
                            help="PyTorch execution device (only for --platform pytorch)")
    arg_parser.add_argument("-mn", "--model_name", type=str, default="superpoint",
                            help="Model name for PyTorch (only for --platform pytorch)")
    arg_parser.add_argument("-s", "--save-path", type=Path, default=Path("/results"),
                            help="Output directory path for profiling files (Chrome trace, ETDump)")
    arg_parser.add_argument("-dbs", "--debug-buffer-size", type=int, default=int(5e7),
                            help="ETDump debug buffer size in bytes (only for --platform executorch)")

    return arg_parser.parse_args()


def main():
    args = parser()

    logger.info(f"Starting profiling for: {args.platform}")
    logger.info(f"Model: {args.modelpath}, Shape: {tuple(args.shape)}")
    logger.info(f"Iterations: {args.iterations}, Warmup: {args.warmup}")

    try:
        profiler = ForwardProfiler(
            input_shape=tuple(args.shape),
            warmup=args.warmup,
            iterations=args.iterations
        )

        results = {}

        if args.platform == "opencv":
            results = profiler.profile_opencv_dnn(args.modelpath)
            logger.info(f"Forward Time: {results['total_time_ms']:.2f} ms")
            logger.info(f"{'Layer Name':<75} {'Type':<25} {'Time (ms)':>10}")
            logger.info("-" * 112)
            for layer in results['layers_profile']:
                if layer['time_ms'] > 0.1:
                    logger.info(f"{layer['name']:<75} {layer['type']:<25} {layer['time_ms']:>10.2f}")

        elif args.platform == "executorch":
            results = profiler.profile_executorch(
                model_path=args.modelpath,
                save_path=args.save_path,
                debug_buffer_size=args.debug_buffer_size
            )
            logger.info(f"Forward Time: {results['total_time_ms']:.2f} ms")

        if 'trace_file' in results:
            logger.info(f"Trace saved to: {results['trace_file']}")
        if 'etdump_file' in results:
            logger.info(f"ETDump saved to: {results['etdump_file']}")
        if 'debug_file' in results:
            logger.info(f"Debug binary: {results['debug_file']}")

        return 0

    except Exception as e:
        logger.error(f"Profiling failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main() or 0)
