#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/../.."

TUNING_CKPT=${TUNING_CKPT:-pretrained/rt-detrv2/rtdetrv2_r50vd_6x_coco_ema.pth}

if [ ! -f "$TUNING_CKPT" ]; then
    echo "Error: COCO pretrained checkpoint not found: $TUNING_CKPT"
    exit 1
fi

export TUNING_CKPT

bash experiments/scripts/run_train.sh \
    experiments/configs/r50_freeze_full.yml \
    r50_freeze_full \
    "$@"
