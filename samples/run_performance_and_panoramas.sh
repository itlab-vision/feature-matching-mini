#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: samples/run_performance_and_panoramas.sh -i IMAGES_DIR [options]

Options:
  -i, --images-dir DIR   Directory with panorama images (required)
  -o, --output-dir DIR   Directory for all results (default: results_6_11)
  -m, --models-dir DIR   Directory with converted models (default: OUTPUT_DIR/models)
  -n, --iterations N     Performance iterations per combination (default: 10)
  -h, --help             Show this help

The script uses the currently active Python interpreter.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RESULTS_DIR="results_6_11"
MODELS_DIR=""
IMAGES_DIR=""
ITERATIONS=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--images-dir)
      [[ $# -ge 2 ]] || { echo "$1 requires a directory" >&2; exit 2; }
      IMAGES_DIR="$2"
      shift 2
      ;;
    -o|--output-dir)
      [[ $# -ge 2 ]] || { echo "$1 requires a directory" >&2; exit 2; }
      RESULTS_DIR="$2"
      shift 2
      ;;
    -m|--models-dir)
      [[ $# -ge 2 ]] || { echo "$1 requires a directory" >&2; exit 2; }
      MODELS_DIR="$2"
      shift 2
      ;;
    -n|--iterations)
      [[ $# -ge 2 ]] || { echo "$1 requires a number" >&2; exit 2; }
      ITERATIONS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$IMAGES_DIR" ]]; then
  echo "Panorama images directory is required; use -i or --images-dir" >&2
  usage >&2
  exit 2
fi
if ! [[ "$ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Iterations must be a positive integer: $ITERATIONS" >&2
  exit 2
fi
if ! command -v python >/dev/null 2>&1; then
  echo "No active Python interpreter found" >&2
  exit 1
fi
if [[ ! -d "$IMAGES_DIR" ]]; then
  echo "Images directory does not exist: $IMAGES_DIR" >&2
  exit 1
fi
MODELS_DIR="${MODELS_DIR:-$RESULTS_DIR/models}"
if [[ ! -d "$MODELS_DIR" ]]; then
  echo "Models directory does not exist: $MODELS_DIR" >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"
RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"
MODELS_DIR="$(cd "$MODELS_DIR" && pwd)"
IMAGES_DIR="$(cd "$IMAGES_DIR" && pwd)"
mkdir -p "$RESULTS_DIR"/{configs,panoramas,reports,logs}

echo "Running performance benchmark..."
python samples/performance_benchmark_table.py \
  -i1 test_data/20251126_131827.jpg \
  -i2 test_data/20251126_131833.jpg \
  -o "$RESULTS_DIR/performance.csv" \
  -d cpu \
  -n "$ITERATIONS" \
  --methods sift_lightglue disk_lightglue aliked_lightglue doghardnet_lightglue superpoint_lightglue \
  --backends torch onnx opencv executorch \
  --model-dir "$MODELS_DIR" \
  --num-keypoints 512

echo "Generating panorama configs..."
export ROOT_DIR RESULTS_DIR MODELS_DIR IMAGES_DIR
python - <<'PY'
from copy import deepcopy
import os
from pathlib import Path

import yaml

root = Path(os.environ["ROOT_DIR"])
results_dir = Path(os.environ["RESULTS_DIR"])
models_dir = Path(os.environ["MODELS_DIR"])
images_dir = Path(os.environ["IMAGES_DIR"])
base = yaml.safe_load((root / "samples/configs/detailed_stitching_src_config.yaml").read_text())[0]
methods = ["sift_lightglue", "disk_lightglue", "aliked_lightglue", "doghardnet_lightglue", "superpoint_lightglue"]
backends = ["torch", "onnx", "opencv", "executorch"]
input_shapes = {
    "disk_lightglue": [1, 3, 1024, 1024],
    "superpoint_lightglue": [1, 3, 480, 640],
    "aliked_lightglue": [1, 3, 640, 640],
}

for method in methods:
    feature = method.replace("_lightglue", "")
    for backend in backends:
        extractor_backend = "torch" if method in ("sift_lightglue", "doghardnet_lightglue") else backend
        config = deepcopy(base)
        name = f"{method}_{backend}"
        config["stitching_params"].update({
            "images_dir": str(images_dir),
            "out_file_name": str(results_dir / "panoramas" / f"{name}.jpg"),
            "report_file_name": str(results_dir / "reports" / f"{name}.yaml"),
        })
        config["matching_params"].update({
            "detector": method,
            "descriptor": method,
            "matcher": "lightglue",
            "sequential_matching": True,
            "keypoints_num": 512,
        })

        detector = {"backend": extractor_backend, "nfeatures": 512}
        descriptor = {"backend": extractor_backend, "nfeatures": 512}
        matcher = {"backend": backend}
        if extractor_backend == "torch":
            detector["device"] = "cpu"
            descriptor["device"] = "cpu"
        if method in ("sift_lightglue", "doghardnet_lightglue"):
            detector["threshold"] = 0.0
            descriptor["threshold"] = 0.0
        else:
            detector["threshold"] = -1e9
            descriptor["threshold"] = -1e9

        if extractor_backend in ("onnx", "opencv"):
            if method == "disk_lightglue":
                model_path = str(models_dir / "disk_1024.onnx")
                detector["disk_model_path"] = model_path
                descriptor["disk_model_path"] = model_path
            elif method == "aliked_lightglue":
                model_path = str(models_dir / "aliked_lightglue.onnx")
                detector["aliked_model_path"] = model_path
                descriptor["aliked_model_path"] = model_path
            else:
                model_path = str(models_dir / "superpoint_lightglue.onnx")
                detector["superpoint_model_path"] = model_path
                descriptor["superpoint_model_path"] = model_path
        elif extractor_backend == "executorch":
            model_path = str(models_dir / f"{method}_xnnpack.pte")
            detector.update({
                "executorch_model_path": model_path,
                "input_shape": input_shapes[method],
                "num_keypoints": 512,
            })
            descriptor.update({
                "executorch_model_path": model_path,
                "input_shape": input_shapes[method],
                "num_keypoints": 512,
            })

        if backend == "torch":
            matcher["device"] = "cpu"
        elif backend == "executorch":
            matcher.update({
                "executorch_model_path": str(models_dir / f"lightglue_{feature}_k512_xnnpack.pte"),
                "num_keypoints": 512,
                "normalize_keypoints": True,
            })
        else:
            model_name = f"lightglue_{feature}_opencv.onnx" if backend == "opencv" \
                else f"lightglue_{feature}_k512.onnx"
            matcher.update({
                "lightglue_model_path": str(models_dir / model_name),
                "normalize_keypoints": True,
            })

        config["matching_params"]["src_config"] = {
            "detector": detector,
            "descriptor": descriptor,
            "matcher": matcher,
        }
        output_path = results_dir / "configs" / f"{name}.yaml"
        output_path.write_text(yaml.safe_dump([config], sort_keys=False, allow_unicode=True))
PY

echo "Stitching panoramas..."
for config in "$RESULTS_DIR"/configs/*.yaml; do
  echo "Running $config"
  python samples/panorama_stitching_sample.py --config "$config"
done

echo "Done. Results are in $RESULTS_DIR"
