#!/bin/bash
# E18a: DINOv3 freeze full with PassThroughEncoder — remove HybridEncoder while keeping decoder unchanged
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/../.."

bash experiments/scripts/run_train.sh \
    experiments/configs/dinov3_freeze_full_encoder_free.yml \
    dinov3_freeze_full_encoder_free \
    "$@"