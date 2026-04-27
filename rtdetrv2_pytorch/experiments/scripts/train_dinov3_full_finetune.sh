#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/../.."

bash experiments/scripts/run_train.sh \
    experiments/configs/dinov3_full_finetune.yml \
    dinov3_full_finetune \
    "$@"