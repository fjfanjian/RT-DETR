#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/../.."

TRAIN_ANN_FILE=${TRAIN_ANN_FILE:-experiments/data/visdrone_subsets/train_coco_10pct.json}

if [ ! -f "$TRAIN_ANN_FILE" ]; then
    echo "Error: subset annotation not found: $TRAIN_ANN_FILE"
    echo "Run: bash experiments/scripts/prepare_visdrone_subsets.sh"
    exit 1
fi

export TRAIN_ANN_FILE

TUNING_CKPT=${TUNING_CKPT:-pretrained/rt-detrv2/rtdetrv2_r50vd_6x_coco_ema.pth}

if [ ! -f "$TUNING_CKPT" ]; then
    echo "Error: COCO pretrained checkpoint not found: $TUNING_CKPT"
    exit 1
fi

export TUNING_CKPT

bash experiments/scripts/run_train.sh \
    experiments/configs/r50_freeze_10pct.yml \
    r50_freeze_10pct \
    "$@"
