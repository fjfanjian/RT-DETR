#!/bin/bash
# DINOv3 freeze full with SparseHybridEncoder
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/../.."

bash experiments/scripts/run_train.sh \
    experiments/configs/dinov3_freeze_full_sparse_encoder.yml \
    dinov3_freeze_full_sparse_encoder \
    "$@"
