#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/../.."

bash experiments/scripts/run_train.sh \
    experiments/configs/r50_full.yml \
    r50_full \
    "$@"