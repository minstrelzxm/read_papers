#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available in PATH." >&2
  exit 1
fi

HAD_NOUNSET=0
if [[ $- == *u* ]]; then
  HAD_NOUNSET=1
  set +u
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ./.conda

if [[ "$HAD_NOUNSET" == "1" ]]; then
  set -u
fi

export PYTHONUNBUFFERED=1

cmd=(
  python "$ROOT_DIR/scripts/local/run_native_batch_test.py"
  --count "${COUNT:-10}"
  --output-root "${OUTPUT_ROOT:-$ROOT_DIR/test_folder/native_batch_test}"
  --provider "${PROVIDER:-local}"
  --model "${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
)

if [[ -n "${SEED:-}" ]]; then
  cmd+=(--seed "$SEED")
fi

if [[ -n "${API_KEY:-}" ]]; then
  cmd+=(--api_key "$API_KEY")
fi

if [[ -n "${BASE_URL:-}" ]]; then
  cmd+=(--base_url "$BASE_URL")
fi

cmd+=("$@")
"${cmd[@]}"
