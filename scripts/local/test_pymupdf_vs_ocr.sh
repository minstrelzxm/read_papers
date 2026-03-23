#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ROOT_DIR/.conda"
set -u

python "$ROOT_DIR/scripts/local/compare_pymupdf_vs_ocr.py" --clean "$@"
