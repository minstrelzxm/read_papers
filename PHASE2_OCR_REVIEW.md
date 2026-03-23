# Phase 2 OCR Review

This note tracks the current state of the second development phase: PDF to structured OCR output.

Reviewed files:

- `src/ocr_engine.py`
- `src/pipeline.py`
- `main.py`
- `scripts/local/test_example_paper.sh`
- `src/analyzer.py` where it depends on OCR output layout

Sanity and runtime checks performed during the latest fix cycle:

- `python -m py_compile main.py src/pipeline.py src/ocr_engine.py` passed.
- `flash-attn==2.7.3` was installed successfully inside `./.conda`.
- `OCREngine` now loads with `flash_attention_2` on the project GPU stack.
- A live one-page OCR probe on the example paper produced valid Markdown output.
- A batch orchestration test confirmed the OCR model is loaded once and reused across multiple PDFs in a single run.

## Current Status

Phase 2 is in a much better state than before.

The major runtime and orchestration issues that were blocking development are now fixed:

- The broken `DeepSeek-OCR-2` output made of repeated `<｜begin▁of▁sentence｜>` tokens is fixed.
- The project now runs on an official-style DeepSeek OCR stack in `./.conda`:
  - `torch==2.6.0`
  - `transformers==4.46.3`
  - `tokenizers==0.20.3`
  - `flash-attn==2.7.3`
- OCR runs one page at a time instead of rasterizing an entire PDF up front.
- The OCR model is loaded once per batch run instead of once per PDF.
- Invalid OCR output is rejected instead of being silently accepted.
- The smoke-test path is now documented in code through `scripts/local/test_example_paper.sh`.

The OCR phase is now usable for real development work, not just ad hoc debugging.

## What Is Already Good

- `src/ocr_engine.py` now has a clear responsibility: one OCR engine instance processes one PDF into a stable output layout.
- `src/pipeline.py` now keeps the OCR model resident across multiple papers in the same run, which removes a major source of wasted startup time.
- The page output contract is useful and consistent:
  - `pages/page_n/original.jpg`
  - `pages/page_n/result.mmd`
  - `full_extracted.md`
- OCR logs now stream directly from the active process, so long runs are observable instead of looking stalled.
- The sample smoke-test script is now usable for local validation against the example paper in `test_folder/`.

## Resolved Issues

### 1. Runtime mismatch broke OCR decoding

Status: fixed

What changed:

- The OCR environment was rebuilt to the DeepSeek-compatible stack.
- `flash-attn==2.7.3` is now installed and available.
- `src/ocr_engine.py` now selects `flash_attention_2` when available.

Result:

- `DeepSeek-OCR-2` now returns valid Markdown on the example paper instead of endless special-token output.

### 2. Whole-PDF rasterization caused unnecessary memory risk

Status: fixed

What changed:

- OCR now renders pages one at a time using `convert_from_path(..., first_page=n, last_page=n)`.

Result:

- Stage 2 no longer loads every page image into memory before OCR starts.

### 3. Missing or bogus OCR output could be accepted as success

Status: fixed

What changed:

- `src/ocr_engine.py` now requires `result.mmd` to exist.
- OCR output is validated with `is_valid_ocr_text()`.

Result:

- Broken pages fail loudly instead of silently writing junk into `full_extracted.md`.

### 4. The OCR model was reloaded for every PDF

Status: fixed

What changed:

- `src/pipeline.py` now constructs one `OCREngine` and reuses it across all pending PDFs in the batch.

Result:

- Multi-paper OCR runs no longer pay full model load cost for every paper.

## Remaining Findings

### 1. One failed page still causes the whole paper output to be deleted

Severity: high

References:

- `src/pipeline.py`
- `src/ocr_engine.py`

Issue:

- If OCR fails on one page, the pipeline still removes the full paper output directory through `cleanup_partial_output()`.
- Successfully completed earlier pages are lost.

Why it matters:

- This wastes long OCR runs.
- It makes debugging one bad page expensive.
- It is now the biggest reliability gap in Phase 2.

Recommended change:

- Add a page-level manifest such as `ocr_status.json`.
- Keep successful page outputs.
- Retry only failed pages.
- Make clean deletion opt-in instead of the default recovery path.

### 2. The new in-process batch runner no longer has a hard per-paper timeout

Severity: medium

References:

- `src/pipeline.py`

Issue:

- The old subprocess-based OCR path had a timeout boundary per paper.
- The new in-process runner removed model reload overhead, but it also removed that hard timeout isolation.

Why it matters:

- A hanging OCR call can now stall the whole OCR batch.
- This is the main tradeoff introduced by the performance fix.

Recommended change:

- Move Stage 2 to a persistent worker process with a queue.
- Keep one loaded OCR model per worker, but preserve process-level timeout and recovery behavior.

### 3. Page rendering is still likely the next OCR speed bottleneck

Severity: medium

References:

- `src/ocr_engine.py`

Issue:

- The code now renders one page at a time, which fixed memory pressure.
- It still relies on `pdf2image` for each page render.

Why it matters:

- This is safer than the previous design, but likely slower than an in-process PDF renderer such as PyMuPDF.
- Once model reload overhead is removed, page rendering becomes a more visible part of total OCR time.

Recommended change:

- Benchmark `pdf2image` against a PyMuPDF-based page renderer.
- If PyMuPDF is materially faster, switch Stage 2 rendering to it.

### 4. OCR always runs in a heavy, debug-friendly mode

Severity: medium

References:

- `src/ocr_engine.py`

Issue:

- OCR currently uses:
  - `base_size=1024`
  - `image_size=768`
  - `crop_mode=True`
  - `save_results=True`
- This is good for quality and debugging, but expensive for routine runs.

Why it matters:

- It increases OCR runtime and disk I/O.
- It makes smoke tests slower than necessary.

Recommended change:

- Add explicit OCR profiles such as:
  - `fast`
  - `default`
  - `debug`
- Allow `save_results=False` for non-debug runs.
- Expose image sizing and crop behavior as CLI flags.

### 5. CUDA and local environment assumptions are still implicit

Severity: low

References:

- `src/ocr_engine.py`
- `scripts/local/test_example_paper.sh`

Issue:

- The project now has a working CUDA OCR stack, but the setup expectations are still mostly encoded in scripts and environment state.
- Stage 2 still assumes a valid CUDA device by default.

Why it matters:

- New developers can still hit setup friction even though the core pipeline is fixed.

Recommended change:

- Add a clearer OCR environment section in the README if not already present.
- Add an explicit preflight command for:
  - GPU availability
  - `flash_attn` import
  - model load smoke test

## Suggested Priority Order

### Highest priority

- Preserve successful pages and retry failed pages instead of deleting the whole paper.
- Restore timeout/recovery isolation with a persistent OCR worker model.

### Next priority

- Add OCR fast/default/debug modes.
- Reduce page rendering overhead if profiling shows `pdf2image` is now the main bottleneck.

### Nice to have

- Add an OCR manifest per paper.
- Add a tiny regression corpus and automated smoke tests for Phase 2.
- Add explicit preflight checks for the OCR environment.

## Bottom Line

Phase 2 is no longer blocked by the earlier runtime failure. The OCR stage now runs on the correct DeepSeek stack, uses FlashAttention 2, processes pages incrementally, and avoids reloading the model for every paper.

The remaining OCR work is no longer about basic correctness. It is about hardening and speed:

- recover cleanly from failed pages
- keep timeout isolation without reintroducing model reload overhead
- cut unnecessary rendering and I/O cost

Those are the right next Phase 2 tasks.
