#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SMOKE_MODE="${SMOKE_MODE:-ocr}" bash "$ROOT_DIR/scripts/local/test_example_paper.sh"
