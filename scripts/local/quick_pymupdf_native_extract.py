import argparse
import shutil
from pathlib import Path

import fitz


DEFAULT_OUTPUT_ROOT = Path("test_folder/pymupdf_native_extract")


def default_pdf_paths():
    return sorted(Path("test_folder").glob("*.pdf"))


def extract_pdf_native(pdf_path, output_root):
    doc = fitz.open(pdf_path)
    paper_dir = output_root / pdf_path.stem
    pages_dir = paper_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    combined_text_parts = []
    total_images = 0

    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        page_dir = pages_dir / f"page_{page_index + 1:03d}"
        images_dir = page_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        text = page.get_text("text", sort=True).strip()
        (page_dir / "text.txt").write_text(text, encoding="utf-8")
        combined_text_parts.append(f"\n\n===== PAGE {page_index + 1} =====\n\n{text}")

        seen_xrefs = set()
        for image_index, image_info in enumerate(page.get_images(full=True), start=1):
            xref = image_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            extracted = doc.extract_image(xref)
            ext = extracted.get("ext", "bin")
            image_path = images_dir / f"image_{image_index:02d}_xref_{xref}.{ext}"
            image_path.write_bytes(extracted["image"])
            total_images += 1

    (paper_dir / "document_text.txt").write_text(
        "".join(combined_text_parts).lstrip(),
        encoding="utf-8",
    )

    print(f"{pdf_path.name}: pages={doc.page_count}, embedded_images={total_images}, output={paper_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Quick PyMuPDF extractor for native PDF text and embedded images."
    )
    parser.add_argument(
        "pdfs",
        nargs="*",
        type=Path,
        help="PDF paths to process. Defaults to all PDFs in test_folder/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Directory for extracted files. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the output directory before extracting.",
    )
    args = parser.parse_args()

    pdf_paths = args.pdfs or default_pdf_paths()
    if not pdf_paths:
        raise SystemExit("No PDF files found.")

    if args.clean and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    for pdf_path in pdf_paths:
        extract_pdf_native(pdf_path, args.output_root)


if __name__ == "__main__":
    main()
