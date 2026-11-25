#!/usr/bin/env bash

set -euo pipefail

SHARE_URL="https://cloud.unn.ru/s/2Z7CDPayX9s4Acs"
MODELS_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ARCHIVE_ROOT="Converted_models"
EXPECTED_FILES=(
    "aliked_lightglue.onnx"
    "aliked_lightglue.onnx.data"
    "aliked_lightglue_xnnpack.pte"
    "disk_1024.onnx"
    "disk_lightglue.onnx"
    "disk_lightglue.onnx.data"
    "disk_lightglue_xnnpack.pte"
    "lightglue_aliked_k512.onnx"
    "lightglue_aliked_k512.onnx.data"
    "lightglue_aliked_k512_xnnpack.pte"
    "lightglue_aliked_opencv.onnx"
    "lightglue_disk_k512.onnx"
    "lightglue_disk_k512.onnx.data"
    "lightglue_disk_k512_xnnpack.pte"
    "lightglue_disk_opencv.onnx"
    "lightglue_doghardnet_k512.onnx"
    "lightglue_doghardnet_k512.onnx.data"
    "lightglue_doghardnet_k512_xnnpack.pte"
    "lightglue_doghardnet_opencv.onnx"
    "lightglue_sift_k512.onnx"
    "lightglue_sift_k512.onnx.data"
    "lightglue_sift_k512_xnnpack.pte"
    "lightglue_sift_opencv.onnx"
    "lightglue_superpoint_k512.onnx"
    "lightglue_superpoint_k512.onnx.data"
    "lightglue_superpoint_k512_xnnpack.pte"
    "lightglue_superpoint_opencv.onnx"
    "superpoint_lightglue.onnx"
    "superpoint_lightglue.onnx.data"
    "superpoint_lightglue_xnnpack.pte"
)

for command in curl unzip; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "Required command not found: ${command}" >&2
        exit 1
    fi
done

mkdir -p "${MODELS_DIR}"
MODELS_DIR="$(cd "${MODELS_DIR}" && pwd)"

models_present=true
for file in "${EXPECTED_FILES[@]}"; do
    if [[ ! -s "${MODELS_DIR}/${file}" ]]; then
        models_present=false
        break
    fi
done

if [[ "${models_present}" == true ]]; then
    echo "All models are already available in ${MODELS_DIR}"
    exit 0
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT
ARCHIVE="${WORK_DIR}/models.zip"

echo "Downloading models to ${MODELS_DIR}"
curl --fail --location --silent --show-error \
    --retry 3 --retry-delay 5 --connect-timeout 30 \
    --output "${ARCHIVE}" "${SHARE_URL}/download"
unzip -tq "${ARCHIVE}"
unzip -q "${ARCHIVE}" -d "${WORK_DIR}"

for file in "${EXPECTED_FILES[@]}"; do
    source_file="${WORK_DIR}/${ARCHIVE_ROOT}/${file}"
    if [[ ! -s "${source_file}" ]]; then
        echo "Model is missing from the downloaded archive: ${file}" >&2
        exit 1
    fi
    cp "${source_file}" "${MODELS_DIR}/${file}"
done

echo "Downloaded ${#EXPECTED_FILES[@]} model files"
