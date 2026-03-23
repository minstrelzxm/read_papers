import argparse
import json
import shutil
import sys
from pathlib import Path

import fitz

ROOT_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR_STR = str(ROOT_DIR)
if ROOT_DIR_STR not in sys.path:
    sys.path.insert(0, ROOT_DIR_STR)

from src.utils import DEFAULT_LAYOUT_SAMPLE_PAGES, classify_pdf_layout, extract_pdf_text_blocks


DEFAULT_OUTPUT_ROOT = Path("test_folder/pymupdf_probe")


def default_pdf_paths():
    return sorted(Path("test_folder").glob("*.pdf"))


def sanitize_metadata(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [sanitize_metadata(item) for item in value]
    if isinstance(value, dict):
        return {str(key): sanitize_metadata(item) for key, item in value.items()}
    return str(value)


def extract_page_blocks(page):
    blocks = []
    for block_index, block in enumerate(page.get_text("blocks", sort=True)):
        x0, y0, x1, y1, text, _block_no, block_type = block
        blocks.append(
            {
                "block_index": block_index,
                "block_type": block_type,
                "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
                "text": (text or "").strip(),
            }
        )
    return blocks


def extract_page_images(doc, page, page_dir):
    images_dir = page_dir / "images"
    images_dir.mkdir(exist_ok=True)

    image_entries = []
    seen_xrefs = set()

    for image_index, image_info in enumerate(page.get_images(full=True), start=1):
        xref = image_info[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)

        extracted = doc.extract_image(xref)
        ext = extracted.get("ext", "bin")
        image_filename = f"image_{image_index:02d}_xref_{xref}.{ext}"
        image_path = images_dir / image_filename
        image_path.write_bytes(extracted["image"])

        try:
            image_rects = [
                [round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2)]
                for rect in page.get_image_rects(xref)
            ]
        except Exception:
            image_rects = []

        image_entries.append(
            {
                "image_index": image_index,
                "xref": xref,
                "path": str(image_path),
                "ext": ext,
                "width": extracted.get("width"),
                "height": extracted.get("height"),
                "colorspace": extracted.get("colorspace"),
                "bpc": extracted.get("bpc"),
                "byte_size": len(extracted["image"]),
                "placements": image_rects,
                "source_info": sanitize_metadata(image_info),
            }
        )

    return image_entries


def analyze_pdf(pdf_path, output_root, sample_pages):
    doc = fitz.open(pdf_path)
    paper_dir = output_root / pdf_path.stem
    paper_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = paper_dir / "pages"
    pages_dir.mkdir(exist_ok=True)

    layout_result = classify_pdf_layout(pdf_path, sample_pages=sample_pages)

    combined_text_parts = []
    page_summaries = []

    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        page_dir = pages_dir / f"page_{page_index + 1:03d}"
        page_dir.mkdir(exist_ok=True)

        text = page.get_text("text", sort=True).strip()
        raw_blocks = extract_page_blocks(page)
        filtered_blocks = extract_pdf_text_blocks(page)
        images = extract_page_images(doc, page, page_dir)

        (page_dir / "text.txt").write_text(text, encoding="utf-8")
        (page_dir / "blocks.json").write_text(
            json.dumps(raw_blocks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (page_dir / "filtered_text_blocks.json").write_text(
            json.dumps(filtered_blocks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        combined_text_parts.append(f"\n\n===== PAGE {page_index + 1} =====\n\n{text}")

        page_summary = {
            "page_number": page_index + 1,
            "char_count": len(text),
            "word_count": len(text.split()),
            "raw_block_count": len(raw_blocks),
            "filtered_block_count": len(filtered_blocks),
            "embedded_image_count": len(images),
            "images": images,
        }
        page_summaries.append(page_summary)

        (page_dir / "page_summary.json").write_text(
            json.dumps(page_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    (paper_dir / "document_text.txt").write_text(
        "".join(combined_text_parts).lstrip(),
        encoding="utf-8",
    )

    summary = {
        "pdf_path": str(pdf_path),
        "page_count": doc.page_count,
        "layout_classification": layout_result,
        "total_characters": sum(page["char_count"] for page in page_summaries),
        "total_words": sum(page["word_count"] for page in page_summaries),
        "total_embedded_images": sum(page["embedded_image_count"] for page in page_summaries),
        "pages_with_images": sum(1 for page in page_summaries if page["embedded_image_count"] > 0),
        "page_summaries": page_summaries,
    }

    (paper_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (paper_dir / "report.md").write_text(
        build_markdown_report(pdf_path, summary),
        encoding="utf-8",
    )

    return summary


def build_markdown_report(pdf_path, summary):
    lines = [
        f"# PyMuPDF Extraction Probe: {pdf_path.name}",
        "",
        "## Document Summary",
        "",
        f"- PDF: `{summary['pdf_path']}`",
        f"- Pages: {summary['page_count']}",
        f"- Layout classification: `{summary['layout_classification']['label']}`",
        f"- Layout reason: {summary['layout_classification']['reason']}",
        f"- Total extracted characters: {summary['total_characters']}",
        f"- Total extracted words: {summary['total_words']}",
        f"- Total embedded images: {summary['total_embedded_images']}",
        f"- Pages with embedded images: {summary['pages_with_images']}",
        "",
        "## Page Breakdown",
        "",
        "| Page | Chars | Words | Raw Blocks | Filtered Blocks | Embedded Images |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for page in summary["page_summaries"]:
        lines.append(
            "| "
            f"{page['page_number']} | "
            f"{page['char_count']} | "
            f"{page['word_count']} | "
            f"{page['raw_block_count']} | "
            f"{page['filtered_block_count']} | "
            f"{page['embedded_image_count']} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Text is extracted from the PDF's native text layer, not OCR.",
            "- Images are extracted only if they are embedded image objects in the PDF.",
            "- Figures drawn as vector graphics or composed from multiple PDF objects may not appear as standalone extracted image files.",
            "- Per-page outputs are stored under `pages/page_XXX/` as `text.txt`, `blocks.json`, `filtered_text_blocks.json`, and `images/`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Probe PyMuPDF text and embedded-image extraction for PDFs."
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
        help=f"Directory where extraction outputs are written. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--sample-pages",
        type=int,
        default=DEFAULT_LAYOUT_SAMPLE_PAGES,
        help="Number of pages from the start of the document to use for layout classification.",
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

    all_results = []
    for pdf_path in pdf_paths:
        summary = analyze_pdf(pdf_path, args.output_root, args.sample_pages)
        all_results.append(summary)
        print(f"\nPDF: {pdf_path}")
        print(f"  layout: {summary['layout_classification']['label']}")
        print(f"  pages: {summary['page_count']}")
        print(f"  total chars: {summary['total_characters']}")
        print(f"  total words: {summary['total_words']}")
        print(f"  embedded images: {summary['total_embedded_images']}")
        print(f"  pages with images: {summary['pages_with_images']}")
        print(
            f"  report: {args.output_root / pdf_path.stem / 'report.md'}"
        )

    combined_summary_path = args.output_root / "combined_summary.json"
    combined_summary_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nCombined summary: {combined_summary_path}")


if __name__ == "__main__":
    main()
