import gc
import shutil
from pathlib import Path

from tqdm import tqdm

from src.analyzer import analyze_paper_folder, build_analyzer
from src.ocr_engine import (
    DEFAULT_EXTRACTION_METHOD,
    DEFAULT_OCR_MODEL,
    build_extraction_engine,
    is_ocr_complete,
    paper_output_dir,
)
from src.scraper import download_papers, get_neurips_2025_papers


DEFAULT_OCR_TIMEOUT = 1200


def run_pipeline(
    limit=None,
    skip_download=False,
    skip_ocr=False,
    skip_analysis=False,
    provider="local",
    model_name=None,
    api_key=None,
    base_url=None,
    ocr_timeout=DEFAULT_OCR_TIMEOUT,
    extraction_method=DEFAULT_EXTRACTION_METHOD,
    ocr_model_name=DEFAULT_OCR_MODEL,
    ocr_device="cuda",
):
    ocr_model_name = ocr_model_name or DEFAULT_OCR_MODEL

    base_dir = Path(__file__).resolve().parents[1]
    original_dir = base_dir / "original_papers"
    extracted_dir = base_dir / "extracted_papers"

    original_dir.mkdir(exist_ok=True)
    extracted_dir.mkdir(exist_ok=True)

    if not skip_download:
        print("--- Step 1: Downloading Papers ---")
        papers = get_neurips_2025_papers()
        if limit:
            papers = papers[:limit]

        print(f"Downloading {len(papers)} papers...")
        download_papers(
            papers,
            original_dir,
            max_workers=5,
            max_retries=3,
            failed_output_path=base_dir / "failed_downloads.txt",
        )

    pdf_files = list_pdf_files(original_dir, limit=limit)
    if not pdf_files:
        print("No papers found to process.")
        return

    if not skip_ocr:
        print("--- Step 2: Text/Image Extraction ---")
        pending_ocr_files = [
            pdf_path
            for pdf_path in pdf_files
            if not is_ocr_complete(paper_output_dir(pdf_path, extracted_dir))
        ]
        if pending_ocr_files:
            print(
                f"Loading extraction engine '{extraction_method}' once for "
                f"{len(pending_ocr_files)} pending paper(s)..."
            )
            ocr_engine = build_extraction_engine(
                method=extraction_method,
                model_name=ocr_model_name,
                device=ocr_device,
            )
        else:
            print("All extraction outputs already exist. Skipping Stage 2 engine load.")
            ocr_engine = None
    else:
        ocr_engine = None

    if not skip_ocr and ocr_engine is not None:
        for pdf_path in tqdm(pdf_files, desc="Stage 2 Pipeline"):
            paper_name = pdf_path.stem
            paper_dir = Path(paper_output_dir(pdf_path, extracted_dir))

            if is_ocr_complete(paper_dir):
                continue

            print(f"Running Stage 2 extraction on {paper_name}...")
            success = run_ocr_in_process(
                ocr_engine,
                pdf_path,
                extracted_dir,
                timeout=ocr_timeout,
            )
            if not success:
                cleanup_partial_output(paper_dir)

        ocr_engine = release_model_memory(ocr_engine)

    analyzer = None
    if not skip_analysis:
        print("--- Step 3: Analysis ---")
        analyzer = build_analyzer(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
        )

        for pdf_path in tqdm(pdf_files, desc="Analysis Pipeline"):
            paper_name = pdf_path.stem
            paper_dir = Path(paper_output_dir(pdf_path, extracted_dir))

            if not is_ocr_complete(paper_dir):
                print(f"Cannot analyze {paper_name}: No extracted text.")
                continue

            if is_analysis_complete(paper_dir):
                continue

            print(f"Analyzing {paper_name}...")
            try:
                analyze_paper_folder(paper_dir, analyzer)
            except Exception as exc:
                print(f"Analysis failed for {paper_name}: {exc}")

        analyzer = release_model_memory(analyzer)

    print("Pipeline Completed.")


def list_pdf_files(original_dir, limit=None):
    pdf_files = sorted(Path(original_dir).glob("*.pdf"))
    if limit is not None:
        pdf_files = pdf_files[:limit]
    return pdf_files


def is_analysis_complete(paper_dir):
    report_path = Path(paper_dir) / "analysis_report.md"
    return report_path.exists() and report_path.stat().st_size > 0


def run_ocr_in_process(ocr_engine, pdf_path, extracted_dir, timeout=DEFAULT_OCR_TIMEOUT):
    if ocr_engine is None:
        print("Stage 2 extraction engine is unavailable.")
        return False

    if timeout != DEFAULT_OCR_TIMEOUT and getattr(ocr_engine, "method_name", None) == "ocr":
        print("Per-paper OCR timeout is not enforced in the in-process batch runner.")

    try:
        return bool(ocr_engine.process_pdf(pdf_path, extracted_dir))
    except Exception as exc:
        print(f"Stage 2 extraction failed for {Path(pdf_path).stem}: {exc}")
        return False


def release_model_memory(model_holder):
    if model_holder is None:
        return None

    del model_holder
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return None


def cleanup_partial_output(paper_dir):
    paper_dir = Path(paper_dir)
    if paper_dir.exists():
        shutil.rmtree(paper_dir)
        print(f"  Cleaned up partial output for {paper_dir.name}")
