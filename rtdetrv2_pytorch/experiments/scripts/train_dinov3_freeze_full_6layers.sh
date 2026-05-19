#!/bin/bash
# E14a: DINOv3 freeze full with 6 decoder layers — align decoder depth with R50
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/../.."

bash experiments/scripts/run_train.sh \
    experiments/configs/dinov3_freeze_full_6layers.yml \
    dinov3_freeze_full_6layers \
    "$@"
