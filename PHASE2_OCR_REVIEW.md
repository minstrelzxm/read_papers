# Phase 2 OCR Review

This note focuses on the second development phase: PDF to structured OCR output.

Reviewed files:

- `main.py`
- `src/ocr_engine.py`
- `src/analyzer.py` where it depends on OCR output format
- auxiliary OCR-related scripts: `run_ocr_robust.py` and `trad_ocr.py`

Sanity check:

- `python -m py_compile main.py src/scraper.py src/ocr_engine.py src/analyzer.py run_ocr_robust.py trad_ocr.py` passed.
- I did not find syntax errors.

## What Is Already Good

- OCR is isolated in a subprocess per PDF, which is a pragmatic way to contain CUDA instability.
- The page-level output layout is useful for debugging and for downstream multimodal analysis.
- The project keeps a merged `full_extracted.md` while also retaining page artifacts, which is the right general contract for later stages.

## Findings

### 1. Whole-PDF rasterization is the main technical risk

Severity: high

References:

- `src/ocr_engine.py:54-68`

Issue:

- `convert_from_path(pdf_path, timeout=600)` loads the full PDF into memory before OCR begins.
- For long papers, this means RAM pressure from all rendered page images before the model has processed even page 1.
- The code then keeps those PIL images in memory while also saving each page to disk, so memory usage scales poorly with paper length.

Why it matters:

- This is the most likely reason Stage 2 will become unstable or slow as paper count and page count increase.
- A single very long paper can fail even if per-page OCR would otherwise succeed.

Recommended change:

- Process PDFs page by page instead of rendering the entire document upfront.
- Use `convert_from_path(..., first_page=n, last_page=n)` or another streaming approach.
- Release page objects immediately after each OCR call.

### 2. OCR success is not validated strongly enough

Severity: medium

References:

- `src/ocr_engine.py:96-112`

Issue:

- If DeepSeek-OCR does not create `result.mmd`, the fallback is `str(res)`.
- If `res` is `None`, the page is still treated as successful and the literal text `None` is written into the merged output.
- That means a broken OCR page can be silently accepted as valid Stage 2 output.

Why it matters:

- Silent data corruption is harder to detect than a hard failure.
- Stage 3 can then analyze incomplete or invalid content without any warning.

Recommended change:

- Require `result.mmd` to exist and be non-empty.
- If it is missing, raise a structured error and mark the page or paper as failed.
- Add an explicit validation step before appending text to `full_extracted.md`.

### 3. One page failure currently discards the whole paper

Severity: medium

References:

- `src/ocr_engine.py:114-117`
- `main.py:104-126`

Issue:

- Any page exception is re-raised from the OCR subprocess.
- The orchestrator then removes the entire paper output directory.
- Successful pages are lost even if only one page failed.

Why it matters:

- This increases rerun cost for long papers.
- It prevents partial recovery and makes debugging specific bad pages harder.

Recommended change:

- Keep a page-level manifest or checkpoint file.
- Retry failed pages independently.
- Preserve successful page outputs unless the user explicitly requests a clean rerun.

### 4. OCR logs are captured, not streamed

Severity: medium

References:

- `main.py:100-107`

Issue:

- `subprocess.run(..., capture_output=True, ...)` buffers all OCR stdout and stderr in memory.
- Developers do not see live OCR progress for a paper.
- Large failure logs can also consume memory unnecessarily.

Why it matters:

- It makes long OCR runs feel stalled.
- It complicates debugging on large batches or GPU failures.

Recommended change:

- Stream stdout and stderr to the console or to per-paper log files.
- Keep only a short error summary in the main orchestrator.

### 5. The OCR workflow has too many overlapping entrypoints

Severity: medium

References:

- `main.py`
- `src/ocr_engine.py:126-176`
- `run_ocr_robust.py`
- `trad_ocr.py`

Issue:

- There are several ways to run Stage 2, but they are not clearly separated into production, debug, and experimental paths.
- `main.py` is the real orchestrator, `src/ocr_engine.py` also has a batch mode, `run_ocr_robust.py` restarts OCR, and `trad_ocr.py` is a comparison script.

Why it matters:

- New contributors will have trouble knowing which entrypoint is authoritative.
- Maintenance cost grows because behavior can drift between scripts.

Recommended change:

- Keep one supported OCR entrypoint.
- Move legacy or experiment scripts into a `scripts/` or `experiments/` directory.
- Document clearly which path is used in production.

### 6. The OCR phase is not friendly to fresh-checkout developer workflows

Severity: low

References:

- `main.py:47-49`

Issue:

- When contributors use `--skip-download`, `main.py` immediately lists `original_papers/`.
- On a fresh checkout without that directory, the pipeline will fail before OCR starts.

Why it matters:

- This is not a Stage 2 algorithm issue, but it does slow down OCR development and testing.
- It makes the common workflow of dropping a few local PDFs into the repo less obvious than it should be.

Recommended change:

- Create `original_papers/` if it is missing.
- Print a clearer message when `--skip-download` is used without input PDFs.

### 7. The OCR stage assumes CUDA without an explicit capability check

Severity: low

References:

- `src/ocr_engine.py:16-21`
- `src/ocr_engine.py:36-40`

Issue:

- `OCREngine` defaults to `device='cuda'` and then moves the model onto that device.
- If a contributor does not have a compatible GPU setup, failure happens late and without a very explicit preflight check.

Why it matters:

- It increases setup friction for contributors who are trying to debug logic, layout, or output contracts on non-GPU machines.

Recommended change:

- Check `torch.cuda.is_available()` before model load.
- Fail with a direct setup message or support a slower CPU fallback for development.

## Suggested Priority Order

### Highest priority

- Stream PDF pages instead of loading the entire document at once.
- Validate OCR success strictly instead of accepting missing `result.mmd` files.
- Preserve successful pages and retry only failed pages.

### Next priority

- Stream OCR logs in real time.
- Simplify the number of OCR entrypoints.
- Improve fresh-checkout developer ergonomics.

### Nice to have

- Add a small regression test corpus with 2 or 3 PDFs.
- Add a machine-readable OCR manifest per paper.
- Add explicit runtime/config flags for OCR-only development work.

## Bottom Line

The current Stage 2 design is workable for experimentation, but it is still closer to a research prototype than a stable developer platform. The biggest risks are memory behavior, silent acceptance of bad OCR output, and rerun cost when a single page fails. Those are the first areas I would tighten before building more functionality on top of this phase.
