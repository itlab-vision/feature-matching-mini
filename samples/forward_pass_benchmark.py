import argparse
import logging
import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # noqa: E402

from src.algorithms import (DNN_DETECTORS, DNN_DESCRIPTORS, DNN_MATCHERS, OPENCV_DESCRIPTORS,  # noqa: E402
                            DESCRIPTOR_MATCHER_COMPATIBILITY)
from src.detectors import Detector  # noqa: E402
from src.descriptors import Descriptor  # noqa: E402
from src.matchers import Matcher  # noqa: E402
from src.super_point import SuperPoint  # noqa: E402, F401
from src.lightglue_matcher import LightGlue  # noqa: E402, F401
from src.lightglue_pipeline import LightGlueFeatureExtractor  # noqa: E402, F401
from src.opencv_dnn_extractors import ALIKEDOpenCV, DISKOpenCV  # noqa: E402, F401
from src.opencv_dnn_matchers import LightGlueOpenCVMatcher  # noqa: E402, F401
from src.tfeat_descriptor import TFeat  # noqa: E402, F401
from src.hardnet_descriptor import HardNet  # noqa: E402, F401
from src.xfeat import XFeat  # noqa: E402, F401
from src.super_glue import SuperGlueMatcher  # noqa: E402, F401
from src.d2net import D2Net  # noqa: E402, F401
from src.r2d2 import R2D2  # noqa: E402, F401
from src.loftr import LoFTR  # noqa: E402, F401
from src.roma import RoMa  # noqa: E402, F401
from samples.forward_profiler import ForwardProfiler  # noqa: E402


logging.basicConfig(level=logging.INFO, format='[ %(levelname)s ] %(message)s')
logger = logging.getLogger("ForwardPassBenchmark")


def parser():
    arg_parser = argparse.ArgumentParser(
        description="Forward pass analysis for OpenCV DNN, PyTorch and ExecuTorch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument("-p", "--platform", required=True,
                            choices=["opencv", "pytorch", "executorch"], help="Target inference platform")
    arg_parser.add_argument("-m", "--modelpath", type=str,
                            help="Path to model file (.onnx/.pte)")
    arg_parser.add_argument("-ni", "--names-inputs", nargs="+", type=str,
                            default=["image"],
                            help="Names for input layers. "
                                 "Use dot notation for nested dictionaries (e.g., 'image0.keypoints').")
    arg_parser.add_argument("-sh", "--shape", nargs="+", type=int,
                            default=[1, 3, 640, 640],
                            help="Input tensor shapes. Use 0 as separator between different inputs. "
                                 "Example: '1 3 640 640' for single input, "
                                 "'1 500 2 0 1 500 128' for two inputs.")
    arg_parser.add_argument("-w", "--warmup", type=int, default=5,
                            help="Number of warmup iterations before measurement")
    arg_parser.add_argument("-n", "--iterations", type=int, default=10,
                            help="Number of measured iterations")
    arg_parser.add_argument("-s", "--save-path", type=Path, default=Path("results"),
                            help="Output directory path for profiling files")

    arg_parser.add_argument("-d", "--device", type=str, default="cpu",
                            choices=["cpu", "cuda"],
                            help="PyTorch execution device (only for --platform pytorch)")
    arg_parser.add_argument("-mn", "--model_name", type=str, default="superpoint",
                            help="Model name for PyTorch (only for --platform pytorch)")
    arg_parser.add_argument("-dbs", "--debug-buffer-size", type=int, default=int(5e7),
                            help="ETDump debug buffer size in bytes (only for --platform executorch)")
    arg_parser.add_argument("-ud", "--use-dict", action="store_true",
                            help="Force input data to be passed as a dictionary {name: tensor} "
                                 "(only for --platform pytorch")

    return arg_parser.parse_args()


def get_first_compatible_descriptor(matcher_name):
    for desc, matchers in DESCRIPTOR_MATCHER_COMPATIBILITY.items():
        if matcher_name in matchers:
            return desc
    raise ValueError(f"No compatible descriptor found for matcher '{matcher_name}'")


def get_model(model_wrapper):
    if hasattr(model_wrapper, '_matcher'):
        net = model_wrapper._matcher
    elif hasattr(model_wrapper, '_extractor'):
        net = model_wrapper._extractor
    elif hasattr(model_wrapper, 'model'):
        net = model_wrapper.model
    elif hasattr(model_wrapper, '_model'):
        net = model_wrapper._model
    else:
        net = model_wrapper

    if hasattr(net, 'net') and isinstance(net.net, torch.nn.Module):
        return net.net

    return net


def opencv_benchmark(profiler, modelpath, save_path):
    results = profiler.profile_opencv_dnn(modelpath, save_path)
    logger.info(f"Forward Time: {results['total_time_ms']:.2f} ms")
    logger.info(f"{'Layer Name':<75} {'Type':<25} {'Time (ms)':>10}")
    logger.info("-" * 112)
    for layer in results['layers_profile']:
        logger.info(f"{layer['name']:<75} {layer['type']:<25} {layer['time_ms']:>10.2f}")

    return results


def executorch_benchmark(profiler, modelpath, save_path, debug_buffer_size):
    results = profiler.profile_executorch(
        model_path=modelpath,
        save_path=save_path,
        debug_buffer_size=debug_buffer_size
    )
    logger.info(f"Forward Time: {results['total_time_ms']:.2f} ms")

    return results


def pytorch_benchmark(profiler, model_name, device, save_path, use_dict):
    config = {'detector': {}, 'descriptor': {}, 'matcher': {'mode': 'simple'}, 'preprocessor': {}}
    if model_name in DNN_DETECTORS:
        model_wrapper = Detector.create(detector_name=model_name, logger=logger,
                                        config=config)

    elif model_name in DNN_DESCRIPTORS or model_name in OPENCV_DESCRIPTORS:
        model_wrapper = Descriptor.create(descriptor_name=model_name, logger=logger,
                                          config=config)

    elif model_name in DNN_MATCHERS:
        compatible_desc_name = get_first_compatible_descriptor(model_name)
        desc_obj = Descriptor.create(descriptor_name=compatible_desc_name, logger=logger,
                                     config=config)
        model_wrapper = Matcher.create(matcher_name=model_name, descriptor_name=desc_obj,
                                       logger=logger, config=config)

    else:
        raise ValueError(
            f"Model '{model_name}' is not supported on PyTorch platform. "
        )

    model = get_model(model_wrapper)
    results = profiler.profile_pytorch(
        model=model,
        model_name=model_name,
        device=device,
        save_path=save_path,
        use_dict=use_dict)

    logger.info(f"Forward Time: {results['total_time_ms']:.2f} ms")
    logger.info(results['trace_table'])

    return results


def main():
    args = parser()

    logger.info(f"Starting profiling for: {args.platform}")
    logger.info(f"Model: {args.modelpath if args.modelpath else args.model_name}")
    logger.info(f"Input layers: {args.names_inputs}")
    try:
        profiler = ForwardProfiler(
            input_layers=args.names_inputs,
            input_shape=args.shape,
            warmup=args.warmup,
            iterations=args.iterations
        )
        results = {}

        logger.info(f"Iterations: {args.iterations}, Warmup: {args.warmup}")

        if args.platform == "opencv":
            results = opencv_benchmark(profiler, args.modelpath, args.save_path)

        elif args.platform == "executorch":
            results = executorch_benchmark(profiler, args.modelpath, args.save_path,
                                           args.debug_buffer_size)

        else:
            results = pytorch_benchmark(profiler, args.model_name, args.device,
                                        args.save_path, args.use_dict)

        if 'trace_file' in results:
            logger.info(f"Trace saved to: {results['trace_file']}")
        if 'etdump_file' in results:
            logger.info(f"ETDump saved to: {results['etdump_file']}")
        if 'debug_file' in results:
            logger.info(f"Debug binary saved to: {results['debug_file']}")

        return 0

    except Exception as e:
        logger.error(f"Profiling failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main() or 0)
