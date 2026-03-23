import argparse
from pathlib import Path

from PyPDF2 import PdfReader


def main():
    parser = argparse.ArgumentParser(description="Simple PyPDF2 text extraction experiment")
    parser.add_argument("pdf_path", help="Path to the input PDF")
    parser.add_argument("output_path", help="Path to the output markdown file")
    args = parser.parse_args()

    reader = PdfReader(args.pdf_path)
    print(f"Found {len(reader.pages)} pages")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file_handle:
        for index, page in enumerate(reader.pages):
            text = page.extract_text()
            print(f"Page {index}: extracted {len(text or '')} characters")
            file_handle.write(text or "")
            file_handle.write("\n")


if __name__ == "__main__":
    main()
