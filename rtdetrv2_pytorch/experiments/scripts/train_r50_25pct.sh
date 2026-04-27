#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/../.."

TRAIN_ANN_FILE=${TRAIN_ANN_FILE:-experiments/data/visdrone_subsets/train_coco_25pct.json}

if [ ! -f "$TRAIN_ANN_FILE" ]; then
    echo "Error: subset annotation not found: $TRAIN_ANN_FILE"
    echo "Run: bash experiments/scripts/prepare_visdrone_subsets.sh"
    exit 1
fi

export TRAIN_ANN_FILE

bash experiments/scripts/run_train.sh \
    experiments/configs/r50_25pct.yml \
    r50_25pct \
    "$@"