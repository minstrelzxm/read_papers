import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR_STR = str(ROOT_DIR)
if ROOT_DIR_STR not in sys.path:
    sys.path.insert(0, ROOT_DIR_STR)

from src.utils import DEFAULT_LAYOUT_SAMPLE_PAGES, classify_pdf_layout


def default_pdf_paths():
    test_dir = Path("test_folder")
    return sorted(test_dir.glob("*.pdf"))


def main():
    parser = argparse.ArgumentParser(
        description="Classify PDFs as single-column or double-column using PyMuPDF text blocks."
    )
    parser.add_argument(
        "pdfs",
        nargs="*",
        type=Path,
        help="PDF paths to inspect. Defaults to all PDFs in test_folder/.",
    )
    parser.add_argument(
        "--sample-pages",
        type=int,
        default=DEFAULT_LAYOUT_SAMPLE_PAGES,
        help="Number of pages from the start of the document to sample.",
    )
    args = parser.parse_args()

    pdf_paths = args.pdfs or default_pdf_paths()
    if not pdf_paths:
        raise SystemExit("No PDF files found.")

    for pdf_path in pdf_paths:
        result = classify_pdf_layout(pdf_path, sample_pages=args.sample_pages)
        print(f"\nPDF: {pdf_path}")
        print(f"Document layout: {result['label']} ({result['reason']})")
        for page_result in result["page_results"]:
            print(
                "  "
                f"page {page_result['page_number']}: {page_result['label']}"
                f" | blocks={page_result['block_count']}"
                f" | wide={page_result['wide_center_blocks']}"
                f" | left={page_result['left_column_blocks']}"
                f" | right={page_result['right_column_blocks']}"
                f" | {page_result['reason']}"
            )


if __name__ == "__main__":
    main()
