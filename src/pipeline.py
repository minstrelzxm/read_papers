import shutil
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

from src.analyzer import analyze_paper_folder, build_analyzer
from src.ocr_engine import is_ocr_complete, paper_output_dir
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
):
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
        print("--- Step 2: OCR Processing ---")

    analyzer = None
    if not skip_analysis:
        print("--- Step 3: Analysis ---")
        analyzer = build_analyzer(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
        )

    for pdf_path in tqdm(pdf_files, desc="Processing Pipeline"):
        paper_name = pdf_path.stem
        paper_dir = Path(paper_output_dir(pdf_path, extracted_dir))

        if not skip_ocr and not is_ocr_complete(paper_dir):
            print(f"Running OCR on {paper_name}...")
            success = run_ocr_subprocess(
                pdf_path,
                extracted_dir,
                timeout=ocr_timeout,
            )
            if not success:
                cleanup_partial_output(paper_dir)
                continue

        if skip_analysis or analyzer is None:
            continue

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

    print("Pipeline Completed.")


def list_pdf_files(original_dir, limit=None):
    pdf_files = sorted(Path(original_dir).glob("*.pdf"))
    if limit is not None:
        pdf_files = pdf_files[:limit]
    return pdf_files


def is_analysis_complete(paper_dir):
    report_path = Path(paper_dir) / "analysis_report.md"
    return report_path.exists() and report_path.stat().st_size > 0


def run_ocr_subprocess(pdf_path, extracted_dir, timeout=DEFAULT_OCR_TIMEOUT):
    ocr_script = Path(__file__).resolve().with_name("ocr_engine.py")
    cmd = [
        sys.executable,
        str(ocr_script),
        str(pdf_path),
        str(extracted_dir),
    ]

    try:
        result = subprocess.run(cmd, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        print(f"OCR Timed out for {Path(pdf_path).stem}")
        return False
    except Exception as exc:
        print(f"Error launching OCR subprocess for {Path(pdf_path).stem}: {exc}")
        return False

    if result.returncode != 0:
        print(f"OCR failed for {Path(pdf_path).stem}")
        return False

    return True


def cleanup_partial_output(paper_dir):
    paper_dir = Path(paper_dir)
    if paper_dir.exists():
        shutil.rmtree(paper_dir)
        print(f"  Cleaned up partial output for {paper_dir.name}")
