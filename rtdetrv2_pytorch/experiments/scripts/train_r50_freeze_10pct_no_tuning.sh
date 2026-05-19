#!/bin/bash
# E11: r50_freeze_10pct without COCO tuning — fair comparison with dinov3_freeze_10pct
# Only ImageNet pretrained backbone; encoder/decoder from scratch.
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

# NOTE: No TUNING_CKPT — encoder/decoder start from scratch (fair comparison with DINOv3)
bash experiments/scripts/run_train.sh \
    experiments/configs/r50_freeze_10pct_no_tuning.yml \
    r50_freeze_10pct_no_tuning \
    "$@"
