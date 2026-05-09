#!/bin/bash

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: bash experiments/scripts/run_train.sh <config_path> <experiment_name> [extra torch/train args ...]"
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

CONFIG_PATH="$1"
EXPERIMENT_NAME="$2"
shift 2

cd "$ROOT_DIR"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: config file not found: $CONFIG_PATH"
    exit 1
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
VISDRONE_ROOT=${VISDRONE_ROOT:-/home/fj/datasets/visdrone}
DINOV3_WEIGHTS=${DINOV3_WEIGHTS:-/home/fj/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT_DIR/output/experiments}

if [ -z "${NPROC_PER_NODE:-}" ]; then
    IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
    NPROC_PER_NODE=${#GPU_IDS[@]}
fi

OUTPUT_DIR="$OUTPUT_ROOT/$EXPERIMENT_NAME"
LOG_FILE="$OUTPUT_DIR/train.log"
mkdir -p "$OUTPUT_DIR"

if [ ! -d "$VISDRONE_ROOT" ]; then
    echo "Error: VISDRONE_ROOT does not exist: $VISDRONE_ROOT"
    exit 1
fi

UPDATE_ARGS=(
    "train_dataloader.dataset.img_folder=${VISDRONE_ROOT}/VisDrone2019-DET-train/images"
    "val_dataloader.dataset.img_folder=${VISDRONE_ROOT}/VisDrone2019-DET-val/images"
    "val_dataloader.dataset.ann_file=${VISDRONE_ROOT}/val_coco.json"
)

if [ -n "${TRAIN_ANN_FILE:-}" ]; then
    UPDATE_ARGS+=("train_dataloader.dataset.ann_file=${TRAIN_ANN_FILE}")
fi

if grep -q "DINOv3Backbone" "$CONFIG_PATH"; then
    if [ ! -f "$DINOV3_WEIGHTS" ]; then
        echo "Error: DINOv3 weights not found: $DINOV3_WEIGHTS"
        exit 1
    fi
    UPDATE_ARGS+=("DINOv3Backbone.pretrained_path=${DINOV3_WEIGHTS}")
fi

EXTRA_ARGS=()
if [ -n "${TUNING_CKPT:-}" ]; then
    echo "TUNING_CKPT: $TUNING_CKPT"
    EXTRA_ARGS+=(-t "$TUNING_CKPT")
fi

echo "=========================================="
echo "RT-DETRv2 experiment launcher"
echo "=========================================="
echo "Date: $(date)"
echo "Config: $CONFIG_PATH"
echo "Experiment: $EXPERIMENT_NAME"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "VISDRONE_ROOT: $VISDRONE_ROOT"
if grep -q "DINOv3Backbone" "$CONFIG_PATH"; then
    echo "DINOV3_WEIGHTS: $DINOV3_WEIGHTS"
fi
if [ -n "${TRAIN_ANN_FILE:-}" ]; then
    echo "TRAIN_ANN_FILE: $TRAIN_ANN_FILE"
fi
echo "=========================================="

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" torchrun --nproc_per_node="$NPROC_PER_NODE" tools/train.py \
    -c "$CONFIG_PATH" \
    --use-amp \
    --output-dir "$OUTPUT_DIR" \
    -u "${UPDATE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    "$@" 2>&1 | tee "$LOG_FILE"