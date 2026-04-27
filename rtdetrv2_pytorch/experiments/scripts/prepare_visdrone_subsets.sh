#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

VISDRONE_ROOT=${VISDRONE_ROOT:-/home/fj/datasets/visdrone}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/experiments/data/visdrone_subsets}
RATIOS=${RATIOS:-"10 25 50"}
SEED=${SEED:-42}

cd "$ROOT_DIR"

if [ ! -f "$VISDRONE_ROOT/train_coco.json" ]; then
    echo "Error: train_coco.json not found under $VISDRONE_ROOT"
    exit 1
fi

echo "Generating VisDrone subset annotations..."
echo "Input: $VISDRONE_ROOT/train_coco.json"
echo "Output: $OUTPUT_DIR"
echo "Ratios: $RATIOS"

python experiments/tools/prepare_visdrone_subsets.py \
    --input-json "$VISDRONE_ROOT/train_coco.json" \
    --output-dir "$OUTPUT_DIR" \
    --ratios $RATIOS \
    --seed "$SEED" \
    --stratified