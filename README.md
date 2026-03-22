# Read Papers

`read_papers` is a three-stage pipeline for working with research papers:

1. Download PDFs from the internet.
2. Run OCR to extract structured Markdown plus page images.
3. Use an LLM to read the extracted content and generate a report.

The current implementation is wired to NeurIPS 2025 papers hosted on OpenReview, but the repository structure is generic enough that each stage can be replaced independently.

## What Is In Scope Today

- Stage 1 downloads paper PDFs from OpenReview using the NeurIPS 2025 venue ID.
- Stage 2 runs DeepSeek-OCR locally and stores per-page OCR artifacts.
- Stage 3 analyzes the extracted text and images with either a local VLM (`Qwen/Qwen3-VL-4B-Thinking` by default) or an online OpenAI model (`gpt-5` by default).

`main.py` is the primary entrypoint for the full pipeline.

## Pipeline Overview

```text
OpenReview -> original_papers/*.pdf -> OCR page artifacts -> full_extracted.md -> analysis_report.md
```

The orchestration flow in `main.py` is:

1. Fetch paper metadata and download PDFs into `original_papers/`.
2. Run OCR in a separate subprocess per PDF through `src/ocr_engine.py`.
3. Load the OCR output from `extracted_papers/<paper>/pages/` and generate an analysis report.

The OCR stage is intentionally isolated in a subprocess to reduce CUDA/VRAM instability across papers.

## Repository Layout

```text
read_papers/
├── main.py                 # Main pipeline orchestrator
├── requirements.txt        # Python dependencies
├── src/
│   ├── scraper.py          # OpenReview scraping + PDF download logic
│   ├── ocr_engine.py       # DeepSeek-OCR wrapper and OCR CLI
│   └── analyzer.py         # Local/OpenAI analysis logic
├── run_ocr_robust.py       # Legacy/debug OCR restart wrapper
├── trad_ocr.py             # Simple PyPDF2 comparison script
├── testing_bash_prompt.sh  # Local helper script used during development
├── original_papers/        # Generated PDFs (gitignored)
└── extracted_papers/       # Generated OCR + analysis outputs (gitignored)
```

Notes for contributors:

- `run_ocr_robust.py`, `trad_ocr.py`, and `testing_bash_prompt.sh` are auxiliary scripts, not the main supported interface.
- `original_papers/`, `extracted_papers/`, and `failed_downloads.txt` are generated artifacts and are ignored by git.
- There is currently no automated test suite in the repository.

## Runtime Requirements

### System

- Linux environment.
- NVIDIA GPU with CUDA for Stage 2 OCR and local Stage 3 analysis.
- `poppler-utils` for PDF rasterization.
- Enough disk space for downloaded PDFs and extracted page images.

Install Poppler on Ubuntu/Debian:

```bash
sudo apt-get install poppler-utils
```

### Python

- Python 3.12 is the intended version.
- Conda is convenient, but not strictly required by the code.

## Setup

```bash
git clone <repo-url>
cd read_papers
conda create -n read_papers python=3.12
conda activate read_papers
pip install -r requirements.txt
```

Important dependency notes:

- `torch==2.5.1`, `torchvision==0.20.1`, and `transformers==4.46.3` are pinned for the current DeepSeek-OCR setup.
- The first OCR or local-analysis run may download model weights from Hugging Face.
- Online analysis requires OpenAI access. You can pass `--api_key` explicitly or rely on the standard `OPENAI_API_KEY` environment variable.

## Main Commands

### Run The Full Pipeline

```bash
python main.py
```

### Useful Variants

Process only the first 5 papers:

```bash
python main.py --limit 5
```

Download only:

```bash
python main.py --skip-ocr --skip-analysis
```

Run OCR only on PDFs already present in `original_papers/`:

```bash
python main.py --skip-download --skip-analysis
```

Run analysis only on existing OCR output:

```bash
python main.py --skip-download --skip-ocr
```

Use OpenAI for analysis:

```bash
python main.py --skip-download --provider online --model gpt-5 --api_key "sk-..."
```

### CLI Options

| Flag | Meaning |
|---|---|
| `--limit <N>` | Limit how many PDFs are processed. |
| `--skip-download` | Reuse existing files in `original_papers/`. |
| `--skip-ocr` | Skip Stage 2 and reuse existing OCR outputs. |
| `--skip-analysis` | Skip Stage 3. |
| `--provider <local|online>` | Choose the analyzer backend. |
| `--model <name>` | Override the default model name. |
| `--api_key <key>` | OpenAI API key for online mode. |

## Generated Output Layout

For each PDF named `<paper>.pdf`, the pipeline creates:

```text
original_papers/
└── <paper>.pdf

extracted_papers/
└── <paper>/
    ├── full_extracted.md
    ├── analysis_report.md
    └── pages/
        ├── page_0/
        │   ├── original.jpg
        │   ├── result.mmd
        │   ├── result_with_boxes.jpg
        │   └── images/
        │       └── *.jpg
        └── page_1/
            └── ...
```

Meaning of the OCR files:

- `original.jpg`: the rasterized PDF page.
- `result.mmd`: the per-page Markdown returned by DeepSeek-OCR.
- `result_with_boxes.jpg`: OCR visualization with detected regions.
- `images/`: cropped figures or tables when DeepSeek-OCR exports them.
- `full_extracted.md`: concatenation of all `result.mmd` page outputs.
- `analysis_report.md`: Stage 3 output generated by the analyzer.

## Stage Ownership In Code

### Stage 1: Download

- `src/scraper.py`
- Main functions: `get_neurips_2025_papers()`, `download_pdf()`

Current behavior:

- Hard-coded to `NeurIPS.cc/2025/Conference`.
- Uses OpenReview API v2.
- Downloads are parallelized.

### Stage 2: OCR

- `src/ocr_engine.py`
- Main class: `OCREngine`

Current behavior:

- Loads `deepseek-ai/DeepSeek-OCR`.
- Converts a PDF into page images with `pdf2image`.
- Runs OCR page by page.
- Stores both page-level artifacts and a merged `full_extracted.md`.

### Stage 3: Analysis

- `src/analyzer.py`
- Main classes: `PaperContent`, `OpenAIAnalyzer`, `LocalVLMAnalyzer`

Current behavior:

- Reads the OCR output from `pages/page_x/`.
- Uses both page text and extracted images as analyzer input.
- Writes `analysis_report.md` into the paper output directory.

## How To Extend The Project

### Change The Paper Source

Replace the OpenReview-specific logic in `src/scraper.py`.

Good refactor direction:

- keep a source-agnostic `download_pdf()` utility,
- introduce a provider abstraction for paper metadata,
- move venue-specific settings into config.

### Change The OCR Backend

Replace or wrap `OCREngine` in `src/ocr_engine.py`.

Useful compatibility target:

- keep the `extracted_papers/<paper>/pages/page_<n>/` layout stable,
- keep `result.mmd` and `full_extracted.md` semantics stable,
- keep Stage 3 reading from `PaperContent` without needing downstream rewrites.

### Add A New Analyzer

Add a new analyzer class in `src/analyzer.py` following the same `analyze(paper_content)` shape.

Recommended direction:

- keep `PaperContent` as the handoff boundary from OCR to analysis,
- add provider selection in `main.py`,
- avoid changing the OCR directory contract unless necessary.

## Known Operational Caveats

- `main.py` assumes `original_papers/` already exists when using `--skip-download`.
- OCR is expensive: one subprocess is launched per PDF and the OCR model is loaded inside that subprocess.
- The repository already contains large generated data directories locally, but those are not part of the tracked source tree.
- The standalone batch mode in `src/ocr_engine.py` is better treated as a debug path than as the main production interface.

## Troubleshooting

### `poppler` or PDF conversion errors

Install `poppler-utils` and verify `pdftoppm` is available on `PATH`.

### CUDA or transformer compatibility problems

Use the pinned versions in `requirements.txt`. The current OCR implementation is tuned around `transformers==4.46.3` and `torch==2.5.1`.

### Online analysis fails to authenticate

Pass `--api_key` or set `OPENAI_API_KEY` in the environment before running the pipeline.

### OCR-only runs fail on a fresh checkout

Create `original_papers/` and place PDFs there first, or run the download stage once.

## Related Review Notes

See `PHASE2_OCR_REVIEW.md` for a focused review of the current OCR stage, including concrete errors, risks, and recommended improvements for the second development phase.
