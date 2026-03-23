# Read Papers

`read_papers` is a three-stage pipeline for working with research papers:

1. Download PDFs from the internet.
2. Run OCR to extract structured Markdown plus page images.
3. Use an LLM to read the extracted content and generate a report.

The current implementation is wired to NeurIPS 2025 papers hosted on OpenReview, but the repository structure is generic enough that each stage can be replaced independently.

## What Is In Scope Today

- Stage 1 downloads paper PDFs from OpenReview using the NeurIPS 2025 venue ID.
- Stage 2 runs DeepSeek-OCR locally and stores per-page OCR artifacts.
- Stage 3 analyzes the extracted text and images with either:
  - a local transformers model (`Qwen/Qwen2.5-1.5B-Instruct` by default),
  - a local OpenAI-compatible endpoint (`Qwen/Qwen3.5-2B` is supported this way),
  - an OpenAI model (`gpt-5` by default), or
  - a Claude model (`claude-sonnet-4-6` by default).

`main.py` is the primary entrypoint for the full pipeline. Internally, it delegates batch orchestration to `src/pipeline.py`.

## Pipeline Overview

```text
OpenReview -> original_papers/*.pdf -> OCR page artifacts -> full_extracted.md -> analysis_report.md
```

The orchestration flow is implemented by `src/pipeline.py`, which `main.py` calls:

1. Fetch paper metadata and download PDFs into `original_papers/`.
2. Run OCR in a separate subprocess per PDF through `src/ocr_engine.py`.
3. Load the OCR output from `extracted_papers/<paper>/pages/` and generate an analysis report.

The OCR stage is intentionally isolated in a subprocess to reduce CUDA/VRAM instability across papers.

## Repository Layout

```text
read_papers/
├── main.py                 # Thin CLI entrypoint for the full pipeline
├── requirements.txt        # Python dependencies
├── src/
│   ├── pipeline.py         # Batch pipeline orchestration
│   ├── scraper.py          # OpenReview scraping + PDF download logic
│   ├── ocr_engine.py       # DeepSeek-OCR wrapper and single-PDF OCR CLI
│   └── analyzer.py         # Local/OpenAI/Claude analysis logic
├── scripts/
│   ├── debug/
│   │   └── run_ocr_robust.py
│   └── local/
│       ├── test_example_paper.sh
│       └── testing_bash_prompt.sh
├── experiments/
│   └── trad_ocr.py
├── original_papers/        # Generated PDFs (gitignored)
└── extracted_papers/       # Generated OCR + analysis outputs (gitignored)
```

Notes for contributors:

- `scripts/` and `experiments/` contain auxiliary workflows, not the main supported interface.
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
conda create --prefix ./.conda python=3.12
conda activate ./.conda
pip install -r requirements.txt
```

Important dependency notes:

- `torch==2.6.0`, `torchvision==0.21.0`, `transformers==4.46.3`, and `tokenizers==0.20.3` are the currently tested DeepSeek OCR baseline.
- The DeepSeek OCR-2 model card also recommends `flash-attn==2.7.3 --no-build-isolation`, but that package requires a CUDA development environment with `nvcc`. The code falls back to eager attention when `flash-attn` is unavailable.
- The first OCR or local-analysis run may download model weights from Hugging Face.
- OpenAI analysis can use `--api_key` or `OPENAI_API_KEY`.
- Claude analysis can use `--api_key`, `CLAUDE_CODE_API_KEY`, or `ANTHROPIC_API_KEY`.
- Local OpenAI-compatible analysis can use `--base_url`, `LOCAL_OPENAI_BASE_URL`, and `LOCAL_OPENAI_API_KEY`.

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
python main.py --skip-download --provider openai --model gpt-5 --api_key "sk-..."
```

Use Claude for analysis:

```bash
python main.py --skip-download --provider claude --model claude-sonnet-4-6 --api_key "your-claude-key"
```

Use a local OpenAI-compatible endpoint for multimodal Qwen analysis:

```bash
python main.py --skip-download --provider local-openai --model Qwen/Qwen3.5-2B --base_url http://127.0.0.1:8000/v1
```

### CLI Options

| Flag | Meaning |
|---|---|
| `--limit <N>` | Limit how many PDFs are processed. |
| `--skip-download` | Reuse existing files in `original_papers/`. |
| `--skip-ocr` | Skip Stage 2 and reuse existing OCR outputs. |
| `--skip-analysis` | Skip Stage 3. |
| `--provider <local|local-openai|openai|claude|online>` | Choose the analyzer backend. `online` is kept as an alias for `openai`. |
| `--model <name>` | Override the default model name. |
| `--api_key <key>` | API key for OpenAI or Claude online mode. |
| `--base_url <url>` | Base URL for local OpenAI-compatible analysis providers. |

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
- Main functions: `get_neurips_2025_papers()`, `download_pdf()`, `download_papers()`

Current behavior:

- Hard-coded to `NeurIPS.cc/2025/Conference`.
- Uses OpenReview API v2.
- Downloads are parallelized.

### Stage 2: OCR

- `src/ocr_engine.py`
- Main class and helpers: `OCREngine`, `paper_output_dir()`, `is_ocr_complete()`

Current behavior:

- Loads `deepseek-ai/DeepSeek-OCR-2`.
- Reads the PDF page count first, then renders one page at a time with `pdf2image`.
- Runs OCR page by page.
- Stores both page-level artifacts and a merged `full_extracted.md`.

### Stage 3: Analysis

- `src/analyzer.py`
- Main classes and helpers: `PaperContent`, `OpenAIAnalyzer`, `ClaudeAnalyzer`, `LocalVLMAnalyzer`, `build_analyzer()`, `analyze_paper_folder()`

Current behavior:

- Reads the OCR output from `pages/page_x/`.
- Uses a text-only local transformers model by default.
- Local summaries stop at references/appendix/checklist markers so the model focuses on the main paper body.
- For long papers, local text-only analysis now summarizes the paper in page chunks and then synthesizes a final report.
- Supports a local OpenAI-compatible multimodal path for models such as `Qwen/Qwen3.5-2B`.
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
- wire it through `build_analyzer()` in `src/analyzer.py`,
- avoid changing the OCR directory contract unless necessary.

## Development Structure

The current development structure after the reorganization is:

- `main.py` is the only supported batch CLI.
- `src/pipeline.py` owns batch sequencing and subprocess management.
- `src/scraper.py` owns batched download logic.
- `src/ocr_engine.py` owns single-PDF OCR only.
- `src/analyzer.py` owns analyzer creation and per-paper analysis helpers.
- `scripts/` and `experiments/` are intentionally outside the production pipeline surface.

For the detailed keep/merge/move record, see `PIPELINE_FUNCTION_CONSOLIDATION.md`.

## Known Operational Caveats

- OCR is expensive: one subprocess is launched per PDF and the OCR model is loaded inside that subprocess.
- The repository already contains large generated data directories locally, but those are not part of the tracked source tree.
- `src/ocr_engine.py` is now a single-PDF OCR entrypoint. Batch OCR should go through `main.py`.

## Troubleshooting

### `poppler` or PDF conversion errors

Install `poppler-utils` and verify `pdftoppm` is available on `PATH`.

### CUDA or transformer compatibility problems

Use the pinned versions in `requirements.txt`. `DeepSeek-OCR-2` is now tested against the official-style stack in `./.conda` with `transformers==4.46.3`, and the OCR code falls back to eager attention when `flash-attn` is not installable on the host.

### `Qwen/Qwen3.5-2B` fails in local transformers mode

`Qwen/Qwen3.5-2B` is not recognized by `transformers==4.46.3`. Use the default local transformers model `Qwen/Qwen2.5-1.5B-Instruct`, or serve `Qwen/Qwen3.5-2B` behind an OpenAI-compatible endpoint and run with `--provider local-openai`.

### OpenAI analysis fails to authenticate

Pass `--api_key` or set `OPENAI_API_KEY` in the environment before running the pipeline.

### Claude analysis fails to authenticate

Pass `--api_key` or set `CLAUDE_CODE_API_KEY` or `ANTHROPIC_API_KEY` in the environment before running the pipeline.

### OCR-only runs fail on a fresh checkout

Create `original_papers/` and place PDFs there first, or run the download stage once.

## Related Review Notes

See `PHASE2_OCR_REVIEW.md` for a focused review of the current OCR stage, including concrete errors, risks, and recommended improvements for the second development phase.
