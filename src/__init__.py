import sys
from pathlib import Path


def _init_thirdparty_routes():
    project_root = Path(__file__).resolve().parent.parent
    thirdparty_dir = project_root / "thirdparty"

    if not thirdparty_dir.is_dir():
        return

    for item in thirdparty_dir.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            paths = [item]
            paths.extend(subdir for subdir in item.iterdir() if subdir.is_dir() and not subdir.name.startswith("."))

            for path in paths:
                resolved_path = str(path.resolve())
                if resolved_path not in sys.path:
                    sys.path.insert(0, resolved_path)


_init_thirdparty_routes()

# Backend: base classes + registered implementations (torch, opencv)
from src.backend import InferenceAPI, IOAdapter, ModelWrapper  # noqa: E402, F401

# Src base classes (OpenCV built-ins live in the same modules and register
# themselves via __init_subclass__)
from src.detectors import Detector  # noqa: E402, F401
from src.descriptors import Descriptor  # noqa: E402, F401
from src.matchers import Matcher  # noqa: E402, F401

# DNN detector/descriptor/matcher implementations (registered via __init_subclass__/register)
from src.dnn_extractors import DNNFeatureExtractors  # noqa: E402, F401
from src.dnn_matchers import DNNMatcher  # noqa: E402, F401
from src.dnn_pipeline import DNNPipeline  # noqa: E402, F401
from src.opencv_dnn_extractors import ALIKEDOpenCV, DISKOpenCV  # noqa: E402, F401
from src.opencv_dnn_matchers import LightGlueOpenCVMatcher  # noqa: E402, F401
from src.hardnet_descriptor import HardNet  # noqa: E402, F401
from src.tfeat_descriptor import TFeat  # noqa: E402, F401
