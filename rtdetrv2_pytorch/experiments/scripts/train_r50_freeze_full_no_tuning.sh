#!/bin/bash
# E13: r50_freeze_full without COCO tuning — fair comparison with dinov3_freeze_full
# Only ImageNet pretrained backbone; encoder/decoder from scratch.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/../.."

# NOTE: No TUNING_CKPT — encoder/decoder start from scratch (fair comparison with DINOv3)
bash experiments/scripts/run_train.sh \
    experiments/configs/r50_freeze_full_no_tuning.yml \
    r50_freeze_full_no_tuning \
    "$@"
