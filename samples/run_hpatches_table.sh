#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: samples/run_hpatches_table.sh -p HPATCHES_DIR [options]

Options:
  -p, --dataset-path DIR   Path to hpatches-sequences-release (required)
  -o, --output-dir DIR     Directory for results (default: results_6_11)
  -m, --models-dir DIR     Directory with converted models (default: OUTPUT_DIR/models)
  -n, --num-scenes N       Number of scenes to process (default: 116)
  -sbs, --scenes-batch-size N
                           Scenes per batch (default: 4)
  -et, --eval-thresholds T [T ...]
                           Evaluation thresholds in pixels (default: 5.0)
  --exclude-riscv-broken   Exclude DISK and SuperPoint pipelines using PyTorch
  --no-skip                Recompute combinations already present in the CSV
  -h, --help               Show this help

The script uses the currently active Python interpreter.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET_PATH=""
RESULTS_DIR="results_6_11"
MODELS_DIR=""
NUM_SCENES=116
SCENES_BATCH_SIZE=4
EVAL_THRESHOLDS=(5.0)
EXCLUDE_RISCV_BROKEN=0
NO_SKIP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--dataset-path)
      [[ $# -ge 2 ]] || { echo "$1 requires a directory" >&2; exit 2; }
      DATASET_PATH="$2"
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
    -n|--num-scenes)
      [[ $# -ge 2 ]] || { echo "$1 requires a number" >&2; exit 2; }
      NUM_SCENES="$2"
      shift 2
      ;;
    -sbs|--scenes-batch-size)
      [[ $# -ge 2 ]] || { echo "$1 requires a number" >&2; exit 2; }
      SCENES_BATCH_SIZE="$2"
      shift 2
      ;;
    -et|--eval-thresholds)
      shift
      EVAL_THRESHOLDS=()
      while [[ $# -gt 0 && "$1" != -* ]]; do
        EVAL_THRESHOLDS+=("$1")
        shift
      done
      if [[ ${#EVAL_THRESHOLDS[@]} -eq 0 ]]; then
        echo "--eval-thresholds requires at least one number" >&2
        exit 2
      fi
      ;;
    --exclude-riscv-broken)
      EXCLUDE_RISCV_BROKEN=1
      shift
      ;;
    --no-skip)
      NO_SKIP=1
      shift
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

if [[ -z "$DATASET_PATH" ]]; then
  echo "HPatches dataset path is required; use -p or --dataset-path" >&2
  usage >&2
  exit 2
fi
if ! [[ "$NUM_SCENES" =~ ^[1-9][0-9]*$ && "$SCENES_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "Scene counts must be positive integers" >&2
  exit 2
fi
for threshold in "${EVAL_THRESHOLDS[@]}"; do
  if ! [[ "$threshold" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Evaluation threshold must be a non-negative number: $threshold" >&2
    exit 2
  fi
done
if ! command -v python >/dev/null 2>&1; then
  echo "No active Python interpreter found" >&2
  exit 1
fi
if [[ ! -d "$DATASET_PATH" ]]; then
  echo "Dataset directory does not exist: $DATASET_PATH" >&2
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
DATASET_PATH="$(cd "$DATASET_PATH" && pwd)"

ARGS=(
  -p "$DATASET_PATH"
  -o "$RESULTS_DIR/hpatches_results.csv"
  -d cpu
  -n "$NUM_SCENES"
  -sbs "$SCENES_BATCH_SIZE"
  -mp "$MODELS_DIR"
  --num-keypoints 512
  --eval-thresholds "${EVAL_THRESHOLDS[@]}"
)
if [[ "$NO_SKIP" -eq 1 ]]; then
  ARGS+=(--no-skip)
fi

if [[ "$EXCLUDE_RISCV_BROKEN" -eq 1 ]]; then
  python samples/hpatches_table_benchmark.py "${ARGS[@]}" \
    --methods sift_lightglue disk_lightglue aliked_lightglue doghardnet_lightglue superpoint_lightglue \
    --backends onnx opencv executorch
  python samples/hpatches_table_benchmark.py "${ARGS[@]}" \
    --methods sift_lightglue aliked_lightglue doghardnet_lightglue \
    --backends torch
else
  python samples/hpatches_table_benchmark.py "${ARGS[@]}" \
    --methods sift_lightglue disk_lightglue aliked_lightglue doghardnet_lightglue superpoint_lightglue \
    --backends torch onnx opencv executorch
fi
echo "Done. Results are in $RESULTS_DIR/hpatches_results.csv"
