# Pipeline Function Consolidation Notes

This note records the pipeline surface after the reorganization. It replaces the earlier planning document that described what should be merged, moved, or deleted.

## Status

Reorganization status: implemented.

The project now has one authoritative batch pipeline entrypoint:

- `main.py` for the user-facing CLI
- `src/pipeline.py` for batch orchestration

The stage modules now own only stage-specific logic:

- `src/scraper.py` owns Stage 1 download work
- `src/ocr_engine.py` owns Stage 2 single-PDF OCR work
- `src/analyzer.py` owns Stage 3 analyzer construction and per-paper analysis work

Auxiliary scripts were moved out of the top-level production surface:

- `scripts/debug/run_ocr_robust.py`
- `scripts/local/testing_bash_prompt.sh`
- `experiments/trad_ocr.py`

## Final Entry Points

### Production entrypoints

| Entry point | Role | Status |
|---|---|---|
| `main.py` | Main CLI for full pipeline and partial runs | Authoritative |
| `src/ocr_engine.py` | Single-PDF OCR subprocess CLI | Authoritative for Stage 2 subprocess use |
| `src/analyzer.py` | Optional single-paper analysis CLI | Supported secondary CLI |

### Non-production entrypoints

| Entry point | Role | Status |
|---|---|---|
| `scripts/debug/run_ocr_robust.py` | Debug restart wrapper for OCR-only batch runs | Auxiliary |
| `scripts/local/testing_bash_prompt.sh` | Local convenience script | Auxiliary |
| `experiments/trad_ocr.py` | PyPDF2 comparison experiment | Experimental |

## What Was Consolidated

### 1. Download batching

Before:

- `main.py` had its own inline download batching logic.
- `src/scraper.py` had separate batch and retry logic.

Now:

- `src/scraper.download_papers()` is the authoritative Stage 1 batch function.
- `main.py` delegates Stage 1 orchestration through `src/pipeline.run_pipeline()`.

Authoritative functions:

- `src.scraper.get_neurips_2025_papers()`
- `src.scraper.download_pdf()`
- `src.scraper.process_downloads()`
- `src.scraper.download_papers()`

### 2. OCR batch execution

Before:

- `main.py` looped over PDFs and launched OCR subprocesses.
- `src/ocr_engine.py` also had a separate no-argument batch mode.
- `run_ocr_robust.py` wrapped that batch mode.

Now:

- `src/ocr_engine.py` owns only single-PDF OCR plus OCR-completion helpers.
- `src/pipeline.py` owns batch iteration over PDFs.
- `scripts/debug/run_ocr_robust.py` calls the main pipeline in OCR-only mode instead of calling a duplicate OCR batch implementation.

Authoritative functions:

- `src.ocr_engine.paper_output_dir()`
- `src.ocr_engine.is_ocr_complete()`
- `src.ocr_engine.OCREngine.process_pdf()`
- `src.pipeline.run_ocr_subprocess()`

### 3. Analysis setup and report writing

Before:

- Analyzer creation existed in both `main.py` and `src/analyzer.py`.
- Per-paper report generation also existed in both places.

Now:

- `src.analyzer.build_analyzer()` is the authoritative analyzer factory.
- `src.analyzer.analyze_paper_folder()` is the authoritative per-paper analysis runner.
- `src/pipeline.py` calls those shared helpers during batch processing.

Authoritative functions:

- `src.analyzer.build_analyzer()`
- `src.analyzer.analyze_paper_folder()`
- `src.analyzer.PaperContent`
- `src.analyzer.OpenAIAnalyzer`
- `src.analyzer.ClaudeAnalyzer`
- `src.analyzer.LocalVLMAnalyzer`

## Current Function Ownership

### `main.py`

Owns:

- CLI argument parsing only

Delegates to:

- `src.pipeline.run_pipeline()`

### `src/pipeline.py`

Owns:

- batch pipeline sequencing
- PDF discovery
- OCR subprocess launching
- partial-output cleanup
- batch analysis execution

Key functions:

- `run_pipeline()`
- `list_pdf_files()`
- `is_analysis_complete()`
- `run_ocr_subprocess()`
- `cleanup_partial_output()`

### `src/scraper.py`

Owns:

- OpenReview venue fetch
- HTTP session creation
- single-PDF download
- batched download with retries

Key functions:

- `get_neurips_2025_papers()`
- `get_session()`
- `download_pdf()`
- `process_downloads()`
- `download_papers()`

### `src/ocr_engine.py`

Owns:

- OCR model lifecycle
- one-PDF OCR processing
- OCR output-path helpers
- single-PDF CLI for subprocess invocation

Key functions and methods:

- `paper_output_dir()`
- `is_ocr_complete()`
- `OCREngine._load_model()`
- `OCREngine.process_pdf()`

### `src/analyzer.py`

Owns:

- OCR output loading into `PaperContent`
- analyzer factory logic
- provider normalization and API key resolution
- per-paper analysis execution
- optional single-paper CLI

Key functions and classes:

- `PaperContent`
- `OpenAIAnalyzer`
- `ClaudeAnalyzer`
- `LocalVLMAnalyzer`
- `normalize_provider()`
- `build_analyzer()`
- `analyze_paper_folder()`

## Final Keep / Move Decisions

### Kept as production code

- `main.py`
- `src/pipeline.py`
- `src/scraper.py`
- `src/ocr_engine.py`
- `src/analyzer.py`

### Moved out of production surface

- `run_ocr_robust.py` -> `scripts/debug/run_ocr_robust.py`
- `testing_bash_prompt.sh` -> `scripts/local/testing_bash_prompt.sh`
- `trad_ocr.py` -> `experiments/trad_ocr.py`

### Removed duplicate responsibilities

- top-level batch OCR loop inside `src/ocr_engine.py`
- duplicated analyzer setup in `main.py`
- duplicated per-paper analysis write path in `main.py`
- duplicated inline Stage 1 batching in `main.py`

## Current Provider Support

Stage 3 provider support is now:

- `local`
- `openai`
- `claude`
- `online` as a backward-compatible alias for `openai`

Claude key resolution currently supports:

- `--api_key`
- `CLAUDE_CODE_API_KEY`
- `ANTHROPIC_API_KEY`

## Remaining Follow-Up Work

The structural consolidation is done, but a few technical follow-ups still remain if you want to keep hardening the project:

- Stream PDF pages instead of rasterizing the whole PDF into memory at once.
- Add a page-level checkpoint or manifest for OCR reruns.
- Add live integration tests for OpenAI and Claude providers.
- Add a small regression corpus for OCR and analysis.
- Decide whether the optional `src/analyzer.py` CLI should stay long-term or eventually move to `scripts/`.

## Bottom Line

The codebase now has a single batch pipeline path and clear stage ownership. The single-item workers were preserved, duplicated orchestration was collapsed, and the old top-level helper scripts were moved into debug, local, or experiment folders. This file should now be treated as the current development structure reference, not as a future plan.
