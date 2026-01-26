import sys
import os
import argparse

# Add project root to sys.path to allow imports from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ocr_engine import OCREngine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", help="Path to PDF")
    parser.add_argument("output_dir", help="Output directory")
    args = parser.parse_args()
    
    try:
        engine = OCREngine()
        success = engine.process_pdf(args.pdf_path, args.output_dir)
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"OCR Crash: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
