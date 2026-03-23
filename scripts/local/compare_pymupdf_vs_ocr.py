import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR_STR = str(ROOT_DIR)
if ROOT_DIR_STR not in sys.path:
    sys.path.insert(0, ROOT_DIR_STR)

from scripts.local.probe_pymupdf_extraction import analyze_pdf, default_pdf_paths
from src.ocr_engine import DEFAULT_OCR_MODEL, OCREngine, paper_output_dir


DEFAULT_OUTPUT_ROOT = Path("test_folder/pymupdf_vs_ocr")


def count_files(directory):
    if not directory.exists():
        return 0
    return sum(1 for path in directory.rglob("*") if path.is_file())


def count_image_files(directory):
    if not directory.exists():
        return 0
    valid_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    return sum(
        1 for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in valid_suffixes
    )


def page_dir_sort_key(page_dir):
    try:
        return int(page_dir.name.split("_")[-1])
    except ValueError:
        return sys.maxsize


def summarize_ocr_output(pdf_path, ocr_output_root):
    paper_dir = Path(paper_output_dir(pdf_path, ocr_output_root))
    full_text_path = paper_dir / "full_extracted.md"
    pages_dir = paper_dir / "pages"

    if not full_text_path.exists():
        raise FileNotFoundError(f"OCR output missing: {full_text_path}")

    full_text = full_text_path.read_text(encoding="utf-8")
    page_summaries = []

    for page_dir in sorted(pages_dir.glob("page_*"), key=page_dir_sort_key):
        try:
            page_index = int(page_dir.name.split("_")[-1])
        except ValueError:
            continue

        result_path = page_dir / "result.mmd"
        page_text = result_path.read_text(encoding="utf-8") if result_path.exists() else ""
        images_dir = page_dir / "images"

        page_summaries.append(
            {
                "page_number": page_index + 1,
                "char_count": len(page_text),
                "word_count": len(page_text.split()),
                "image_count": count_image_files(images_dir),
                "has_result_mmd": result_path.exists(),
                "images_dir": str(images_dir),
            }
        )

    return {
        "pdf_path": str(pdf_path),
        "paper_dir": str(paper_dir),
        "full_text_path": str(full_text_path),
        "page_count": len(page_summaries),
        "total_characters": len(full_text),
        "total_words": len(full_text.split()),
        "total_extracted_images": sum(page["image_count"] for page in page_summaries),
        "pages_with_images": sum(1 for page in page_summaries if page["image_count"] > 0),
        "page_summaries": page_summaries,
    }


def compare_page_summaries(pymupdf_summary, ocr_summary):
    ocr_pages = {page["page_number"]: page for page in ocr_summary["page_summaries"]}
    comparison_rows = []

    for pymupdf_page in pymupdf_summary["page_summaries"]:
        page_number = pymupdf_page["page_number"]
        ocr_page = ocr_pages.get(page_number, {})
        comparison_rows.append(
            {
                "page_number": page_number,
                "pymupdf_chars": pymupdf_page["char_count"],
                "ocr_chars": ocr_page.get("char_count", 0),
                "char_delta": ocr_page.get("char_count", 0) - pymupdf_page["char_count"],
                "pymupdf_images": pymupdf_page["embedded_image_count"],
                "ocr_images": ocr_page.get("image_count", 0),
                "image_delta": ocr_page.get("image_count", 0) - pymupdf_page["embedded_image_count"],
            }
        )

    return comparison_rows


def build_markdown_report(pdf_path, comparison):
    pymupdf_summary = comparison["pymupdf"]
    ocr_summary = comparison["ocr"]

    lines = [
        f"# PyMuPDF vs DeepSeek-OCR-2: {pdf_path.name}",
        "",
        "## Document Summary",
        "",
        f"- PDF: `{pdf_path}`",
        f"- Pages: {comparison['page_count']}",
        f"- Layout: `{comparison['layout_label']}`",
        f"- PyMuPDF duration: {comparison['durations']['pymupdf_seconds']:.2f}s",
        f"- OCR duration: {comparison['durations']['ocr_seconds']:.2f}s",
        f"- PyMuPDF text chars: {pymupdf_summary['total_characters']}",
        f"- OCR text chars: {ocr_summary['total_characters']}",
        f"- PyMuPDF words: {pymupdf_summary['total_words']}",
        f"- OCR words: {ocr_summary['total_words']}",
        f"- PyMuPDF embedded images: {pymupdf_summary['total_embedded_images']}",
        f"- OCR extracted images: {ocr_summary['total_extracted_images']}",
        f"- PyMuPDF pages with images: {pymupdf_summary['pages_with_images']}",
        f"- OCR pages with images: {ocr_summary['pages_with_images']}",
        "",
        "## Interpretation",
        "",
        "- PyMuPDF reads the PDF's native text layer and embedded image objects.",
        "- DeepSeek-OCR-2 reconstructs page content from rendered page images and may extract figures that are not embedded as standalone PDF images.",
        "- Large differences in image counts are expected on vector-heavy papers.",
        "",
        "## Page Breakdown",
        "",
        "| Page | PyMuPDF Chars | OCR Chars | Char Delta | PyMuPDF Images | OCR Images | Image Delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in comparison["page_comparison"]:
        lines.append(
            "| "
            f"{row['page_number']} | "
            f"{row['pymupdf_chars']} | "
            f"{row['ocr_chars']} | "
            f"{row['char_delta']} | "
            f"{row['pymupdf_images']} | "
            f"{row['ocr_images']} | "
            f"{row['image_delta']} |"
        )

    lines.extend(
        [
            "",
            "## Output Paths",
            "",
            f"- PyMuPDF summary: `{comparison['paths']['pymupdf_summary']}`",
            f"- OCR full text: `{comparison['paths']['ocr_full_text']}`",
            f"- OCR paper dir: `{comparison['paths']['ocr_paper_dir']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def compare_one_pdf(
    pdf_path,
    pymupdf_root,
    ocr_root,
    comparison_root,
    sample_pages,
    engine,
    run_pymupdf,
    run_ocr,
    reuse_ocr,
):
    comparison_root = comparison_root / pdf_path.stem
    comparison_root.mkdir(parents=True, exist_ok=True)

    pymupdf_summary = None
    ocr_summary = None
    pymupdf_seconds = 0.0
    ocr_seconds = 0.0

    if run_pymupdf:
        started = time.perf_counter()
        pymupdf_summary = analyze_pdf(pdf_path, pymupdf_root, sample_pages)
        pymupdf_seconds = time.perf_counter() - started
    else:
        pymupdf_summary_path = pymupdf_root / pdf_path.stem / "summary.json"
        pymupdf_summary = json.loads(pymupdf_summary_path.read_text(encoding="utf-8"))

    if run_ocr:
        paper_dir = Path(paper_output_dir(pdf_path, ocr_root))
        full_text_path = paper_dir / "full_extracted.md"
        should_run_ocr = True
        if reuse_ocr and full_text_path.exists() and full_text_path.stat().st_size > 0:
            should_run_ocr = False

        if should_run_ocr:
            started = time.perf_counter()
            engine.process_pdf(pdf_path, ocr_root)
            ocr_seconds = time.perf_counter() - started
        else:
            ocr_seconds = 0.0

    ocr_summary = summarize_ocr_output(pdf_path, ocr_root)
    page_comparison = compare_page_summaries(pymupdf_summary, ocr_summary)

    comparison = {
        "pdf_path": str(pdf_path),
        "page_count": max(pymupdf_summary["page_count"], ocr_summary["page_count"]),
        "layout_label": pymupdf_summary["layout_classification"]["label"],
        "durations": {
            "pymupdf_seconds": pymupdf_seconds,
            "ocr_seconds": ocr_seconds,
        },
        "pymupdf": pymupdf_summary,
        "ocr": ocr_summary,
        "page_comparison": page_comparison,
        "paths": {
            "pymupdf_summary": str(pymupdf_root / pdf_path.stem / "summary.json"),
            "ocr_full_text": ocr_summary["full_text_path"],
            "ocr_paper_dir": ocr_summary["paper_dir"],
        },
    }

    comparison_json_path = comparison_root / "comparison.json"
    comparison_report_path = comparison_root / "comparison_report.md"
    comparison_json_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    comparison_report_path.write_text(
        build_markdown_report(pdf_path, comparison),
        encoding="utf-8",
    )
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Run PyMuPDF extraction and DeepSeek-OCR-2 on the same PDFs and compare outputs."
    )
    parser.add_argument(
        "pdfs",
        nargs="*",
        type=Path,
        help="PDF paths to inspect. Defaults to all PDFs in test_folder/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Directory where comparison outputs are written. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--sample-pages",
        type=int,
        default=4,
        help="Number of pages from the start of the document to use for layout classification.",
    )
    parser.add_argument(
        "--pymupdf-output-root",
        type=Path,
        help="Directory where PyMuPDF outputs are written or reused. Defaults to <output-root>/pymupdf.",
    )
    parser.add_argument(
        "--ocr-output-root",
        type=Path,
        help="Directory where OCR outputs are written or reused. Defaults to <output-root>/ocr.",
    )
    parser.add_argument(
        "--ocr-model",
        default=DEFAULT_OCR_MODEL,
        help="OCR model name.",
    )
    parser.add_argument(
        "--ocr-device",
        default="cuda",
        help="Torch device for OCR.",
    )
    parser.add_argument(
        "--skip-pymupdf",
        action="store_true",
        help="Do not rerun PyMuPDF extraction. Existing PyMuPDF summary files must already exist.",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Do not rerun OCR. Existing OCR outputs must already exist.",
    )
    parser.add_argument(
        "--reuse-ocr",
        action="store_true",
        help="Reuse existing OCR outputs under the comparison output root when available.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the output root before writing new results.",
    )
    args = parser.parse_args()

    pdf_paths = args.pdfs or default_pdf_paths()
    if not pdf_paths:
        raise SystemExit("No PDF files found.")

    if args.clean and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    pymupdf_root = args.pymupdf_output_root or (args.output_root / "pymupdf")
    ocr_root = args.ocr_output_root or (args.output_root / "ocr")
    comparison_root = args.output_root / "comparison"
    pymupdf_root.mkdir(parents=True, exist_ok=True)
    ocr_root.mkdir(parents=True, exist_ok=True)
    comparison_root.mkdir(parents=True, exist_ok=True)

    run_pymupdf = not args.skip_pymupdf
    run_ocr = not args.skip_ocr

    engine = None
    if run_ocr:
        engine = OCREngine(model_name=args.ocr_model, device=args.ocr_device)

    comparisons = []
    for pdf_path in pdf_paths:
        comparison = compare_one_pdf(
            pdf_path=pdf_path,
            pymupdf_root=pymupdf_root,
            ocr_root=ocr_root,
            comparison_root=comparison_root,
            sample_pages=args.sample_pages,
            engine=engine,
            run_pymupdf=run_pymupdf,
            run_ocr=run_ocr,
            reuse_ocr=args.reuse_ocr,
        )
        comparisons.append(comparison)
        report_path = comparison_root / pdf_path.stem / "comparison_report.md"
        print(f"\nPDF: {pdf_path}")
        print(f"  layout: {comparison['layout_label']}")
        print(f"  pymupdf chars: {comparison['pymupdf']['total_characters']}")
        print(f"  ocr chars: {comparison['ocr']['total_characters']}")
        print(f"  pymupdf images: {comparison['pymupdf']['total_embedded_images']}")
        print(f"  ocr images: {comparison['ocr']['total_extracted_images']}")
        print(f"  report: {report_path}")

    combined_summary_path = args.output_root / "combined_comparison.json"
    combined_report_path = args.output_root / "combined_report.md"
    combined_summary_path.write_text(
        json.dumps(comparisons, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    combined_report_path.write_text(
        "\n".join(
            [
                "# PyMuPDF vs DeepSeek-OCR-2 Combined Report",
                "",
                *[
                    f"- `{Path(item['pdf_path']).name}`: "
                    f"PyMuPDF chars={item['pymupdf']['total_characters']}, "
                    f"OCR chars={item['ocr']['total_characters']}, "
                    f"PyMuPDF images={item['pymupdf']['total_embedded_images']}, "
                    f"OCR images={item['ocr']['total_extracted_images']}, "
                    f"report=`comparison/{Path(item['pdf_path']).stem}/comparison_report.md`"
                    for item in comparisons
                ],
                "",
                "Interpret the image counts carefully: PyMuPDF counts embedded PDF image objects, while OCR counts images the model chose to crop from the rendered page.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"\nCombined summary: {combined_summary_path}")
    print(f"Combined report: {combined_report_path}")


if __name__ == "__main__":
    main()
