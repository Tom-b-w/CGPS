#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/reproduce.sh <data_root> <gpu_id>" >&2
  exit 2
fi

DATA_ROOT="$1"
GPU_ID="$2"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/outputs/main_seed42"
LOG_DIR="${REPO_ROOT}/logs"
DATASETS=(fgvc caltech101 stanford_cars dtd eurosat oxford_flowers food101 oxford_pets sun397 ucf101)

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

for DATASET in "${DATASETS[@]}"; do
  OUTPUT="${OUTPUT_DIR}/${DATASET}_seed42.json"
  LOG="${LOG_DIR}/${DATASET}_seed42.log"
  if [[ -s "${OUTPUT}" ]]; then
    echo "[CGPS] ${DATASET}: existing result, skipping"
    continue
  fi
  echo "[CGPS] ${DATASET}: starting"
  python "${REPO_ROOT}/scripts/evaluate.py" \
    --data-root "${DATA_ROOT}" \
    --datasets "${DATASET}" \
    --seed 42 \
    --output "${OUTPUT}" 2>&1 | tee "${LOG}"
done
