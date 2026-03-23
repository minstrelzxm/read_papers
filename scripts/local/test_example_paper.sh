#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TEST_PDF="${TEST_PDF:-$ROOT_DIR/test_folder/3BASiL_An_Algorithmic_Framework_for_Sparse_plus_LowRank_Compression_of_LLMs_byNNv5Et10.pdf}"
EXTRACTED_DIR="${EXTRACTED_DIR:-$ROOT_DIR/test_folder/ocr_extracted}"
OCR_MODEL="${OCR_MODEL:-deepseek-ai/DeepSeek-OCR-2}"
ANALYZER_PROVIDER="${ANALYZER_PROVIDER:-local}"
LOCAL_MODEL="${LOCAL_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
LOCAL_BASE_URL="${LOCAL_BASE_URL:-http://127.0.0.1:8000/v1}"
SMOKE_MODE="${SMOKE_MODE:-full}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"
REUSE_OCR="${REUSE_OCR:-0}"

if [[ ! -f "$TEST_PDF" ]]; then
  echo "Test PDF not found: $TEST_PDF" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available in PATH." >&2
  exit 1
fi

# Some conda activation hooks assume unset variables are allowed, which
# conflicts with `set -u`. Temporarily relax nounset during activation.
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
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

PAPER_BASENAME="$(basename "$TEST_PDF")"
PAPER_NAME="${PAPER_BASENAME%.pdf}"
OUTPUT_DIR="$EXTRACTED_DIR/$PAPER_NAME"
OCR_OUTPUT_FILE="$OUTPUT_DIR/full_extracted.md"
ANALYSIS_OUTPUT_FILE="$OUTPUT_DIR/analysis_report.md"

mkdir -p "$EXTRACTED_DIR"

case "$SMOKE_MODE" in
  ocr|summary|full)
    ;;
  *)
    echo "Unsupported SMOKE_MODE: $SMOKE_MODE (expected: ocr, summary, or full)" >&2
    exit 1
    ;;
esac

if [[ "$SMOKE_MODE" != "summary" && "$CLEAN_OUTPUT" == "1" && -d "$OUTPUT_DIR" ]]; then
  echo "Removing previous output: $OUTPUT_DIR"
  rm -rf "$OUTPUT_DIR"
fi

echo "Using test PDF: $TEST_PDF"
echo "OCR output dir: $OUTPUT_DIR"
echo "Smoke mode: $SMOKE_MODE"
echo

if [[ "$SMOKE_MODE" == "summary" ]]; then
  if [[ ! -s "$OCR_OUTPUT_FILE" ]]; then
    echo "Summary smoke test requires existing OCR output: $OCR_OUTPUT_FILE" >&2
    exit 1
  fi
  echo "Reusing existing OCR output: $OCR_OUTPUT_FILE"
else
  if [[ "$REUSE_OCR" == "1" && -s "$OCR_OUTPUT_FILE" ]]; then
    echo "Reusing existing OCR output: $OCR_OUTPUT_FILE"
  else
    echo "=== Stage 2: OCR ==="
    python "$ROOT_DIR/src/ocr_engine.py" \
      "$TEST_PDF" \
      "$EXTRACTED_DIR" \
      --model-name "$OCR_MODEL"
  fi

  if [[ ! -s "$OCR_OUTPUT_FILE" ]]; then
    echo "OCR smoke test failed: missing or empty $OCR_OUTPUT_FILE" >&2
    exit 1
  fi

  echo "OCR smoke test passed: $OCR_OUTPUT_FILE"
fi

if [[ "$SMOKE_MODE" == "ocr" ]]; then
  echo
  echo "Smoke test completed."
  echo "OCR output: $OCR_OUTPUT_FILE"
  exit 0
fi

echo
echo "=== Stage 3: Summary ==="
if [[ "$ANALYZER_PROVIDER" == "local-openai" ]]; then
  python "$ROOT_DIR/src/analyzer.py" \
    "$OUTPUT_DIR" \
    --provider "$ANALYZER_PROVIDER" \
    --model "$LOCAL_MODEL" \
    --base_url "$LOCAL_BASE_URL"
else
  python "$ROOT_DIR/src/analyzer.py" \
    "$OUTPUT_DIR" \
    --provider "$ANALYZER_PROVIDER" \
    --model "$LOCAL_MODEL"
fi

if [[ ! -s "$ANALYSIS_OUTPUT_FILE" ]]; then
  echo "Summary smoke test failed: missing or empty $ANALYSIS_OUTPUT_FILE" >&2
  exit 1
fi

echo
echo "Smoke test completed."
echo "OCR output: $OCR_OUTPUT_FILE"
echo "Summary report: $ANALYSIS_OUTPUT_FILE"
