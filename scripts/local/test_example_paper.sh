#!/bin/bash
set -euo pipefail

# Runs the sample paper in test_folder through:
# 1. OCR with DeepSeek-OCR-2
# 2. Local analysis with the provider/model passed in ANALYZER_PROVIDER/LOCAL_MODEL

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_DIR="$ROOT_DIR/test_folder"
TEST_PDF="$ROOT_DIR/test_folder/3BASiL_An_Algorithmic_Framework_for_Sparse_plus_LowRank_Compression_of_LLMs_byNNv5Et10.pdf"
EXTRACTED_DIR="$TEST_DIR/ocr_extracted"
PAPER_BASENAME="$(basename "$TEST_PDF")"
PAPER_NAME="${PAPER_BASENAME%.pdf}"
OUTPUT_DIR="$EXTRACTED_DIR/$PAPER_NAME"

OCR_MODEL="${OCR_MODEL:-deepseek-ai/DeepSeek-OCR-2}"
ANALYZER_PROVIDER="${ANALYZER_PROVIDER:-local}"
LOCAL_MODEL="${LOCAL_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
LOCAL_BASE_URL="${LOCAL_BASE_URL:-http://127.0.0.1:8000/v1}"

mkdir -p "$EXTRACTED_DIR"

if [[ -d "$ROOT_DIR/.conda" ]]; then
  # Make `conda activate` available in non-interactive shells.
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ROOT_DIR/.conda"
fi

echo "Running OCR for $PAPER_BASENAME"
python "$ROOT_DIR/src/ocr_engine.py" \
  "$TEST_PDF" \
  "$EXTRACTED_DIR" \
  --model-name "$OCR_MODEL"

echo "Running local analysis for $PAPER_NAME"
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

echo
echo "Done."
echo "OCR output:    $OUTPUT_DIR/full_extracted.md"
echo "Summary report: $OUTPUT_DIR/analysis_report.md"
