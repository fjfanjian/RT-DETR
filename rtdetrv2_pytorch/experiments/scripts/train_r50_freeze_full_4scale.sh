#!/bin/bash
# E15a: R50 4-scale freeze full — test R50 with stride=[4,8,16,32] similar to DINOv3
# PResNet return_idx=[0,1,2,3] gives native stride=4 output.
# No COCO tuning — encoder/decoder from scratch (fair comparison with DINOv3).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/../.."

bash experiments/scripts/run_train.sh \
    experiments/configs/r50_freeze_full_4scale.yml \
    r50_freeze_full_4scale \
    "$@"
