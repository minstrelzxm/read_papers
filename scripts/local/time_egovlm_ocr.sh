#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TEST_PDF="${TEST_PDF:-$ROOT_DIR/test_folder/EgoVLM__Policy_Optimization_for_Egocentric_Video_Understanding.pdf}"
EXTRACTED_DIR="${EXTRACTED_DIR:-$ROOT_DIR/test_folder/ocr_timing_output}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/test_folder/ocr_timing_logs}"
OCR_MODEL="${OCR_MODEL:-deepseek-ai/DeepSeek-OCR-2}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"

if [[ ! -f "$TEST_PDF" ]]; then
  echo "Test PDF not found: $TEST_PDF" >&2
  exit 1
fi

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
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

PAPER_BASENAME="$(basename "$TEST_PDF")"
PAPER_NAME="${PAPER_BASENAME%.pdf}"
OUTPUT_DIR="$EXTRACTED_DIR/$PAPER_NAME"
OCR_OUTPUT_FILE="$OUTPUT_DIR/full_extracted.md"

mkdir -p "$EXTRACTED_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/${PAPER_NAME}_ocr_$(date +%Y%m%d_%H%M%S).log"

if [[ "$CLEAN_OUTPUT" == "1" && -d "$OUTPUT_DIR" ]]; then
  echo "Removing previous OCR output: $OUTPUT_DIR"
  rm -rf "$OUTPUT_DIR"
fi

echo "Timing OCR for: $TEST_PDF"
echo "OCR output dir: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo

START_TIME="$(date -Iseconds)"
SECONDS=0

{
  echo "=== OCR Timing Run ==="
  echo "Start: $START_TIME"
  echo "PDF: $TEST_PDF"
  echo "Model: $OCR_MODEL"
  echo "Output dir: $OUTPUT_DIR"
  echo

  python "$ROOT_DIR/src/ocr_engine.py" \
    "$TEST_PDF" \
    "$EXTRACTED_DIR" \
    --model-name "$OCR_MODEL"
} 2>&1 | tee "$LOG_FILE"

ELAPSED_SECONDS="$SECONDS"
END_TIME="$(date -Iseconds)"
printf -v ELAPSED_FMT "%02dh:%02dm:%02ds" \
  $((ELAPSED_SECONDS / 3600)) \
  $(((ELAPSED_SECONDS % 3600) / 60)) \
  $((ELAPSED_SECONDS % 60))

if [[ ! -s "$OCR_OUTPUT_FILE" ]]; then
  echo "OCR timing run failed: missing or empty $OCR_OUTPUT_FILE" >&2
  exit 1
fi

{
  echo
  echo "=== OCR Timing Summary ==="
  echo "End: $END_TIME"
  echo "Elapsed seconds: $ELAPSED_SECONDS"
  echo "Elapsed formatted: $ELAPSED_FMT"
  echo "OCR output: $OCR_OUTPUT_FILE"
} | tee -a "$LOG_FILE"

echo
echo "Timing run completed."
echo "Elapsed: $ELAPSED_FMT ($ELAPSED_SECONDS seconds)"
echo "OCR output: $OCR_OUTPUT_FILE"
echo "Log file: $LOG_FILE"
