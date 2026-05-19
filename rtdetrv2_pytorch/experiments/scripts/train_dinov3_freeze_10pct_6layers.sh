#!/bin/bash
# E14c: DINOv3 freeze 10pct with 6 decoder layers — align decoder depth with R50
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

bash experiments/scripts/run_train.sh \
    experiments/configs/dinov3_freeze_10pct_6layers.yml \
    dinov3_freeze_10pct_6layers \
    "$@"
