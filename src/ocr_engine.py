import argparse
import os
import sys

import fitz
from pdf2image import convert_from_path, pdfinfo_from_path
import torch
from transformers import AutoModel, AutoTokenizer, logging as hf_logging

try:
    from src.utils import classify_pdf_layout
except ModuleNotFoundError:
    from utils import classify_pdf_layout


hf_logging.set_verbosity_error()

DEFAULT_EXTRACTION_METHOD = "native"
DEFAULT_OCR_MODEL = "deepseek-ai/DeepSeek-OCR-2"
DEFAULT_OCR_BASE_SIZE = 1024
DEFAULT_OCR_IMAGE_SIZE = 768
INVALID_OCR_TOKEN = "<｜begin▁of▁sentence｜>"
DEFAULT_OCR_PROFILE = "double"
OCR_PROFILES = {
    "single": {
        "base_size": 896,
        "image_size": 704,
        "crop_mode": False,
        "save_results": True,
    },
    "double": {
        "base_size": 1024,
        "image_size": 768,
        "crop_mode": True,
        "save_results": True,
    },
}


def paper_output_dir(pdf_path, output_dir):
    paper_name = os.path.splitext(os.path.basename(pdf_path))[0]
    return os.path.join(os.fspath(output_dir), paper_name)


def is_ocr_complete(paper_dir):
    full_extracted_path = os.path.join(os.fspath(paper_dir), "full_extracted.md")
    return os.path.exists(full_extracted_path) and os.path.getsize(full_extracted_path) > 0


def is_valid_ocr_text(text):
    stripped = text.strip()
    if not stripped:
        return False

    if INVALID_OCR_TOKEN not in stripped:
        return True

    non_special_text = stripped.replace(INVALID_OCR_TOKEN, "").strip()
    return bool(non_special_text)


def _has_flash_attention_2():
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


class PDFNativeExtractor:
    method_name = "native"

    def process_pdf(self, pdf_path, output_dir):
        pdf_path = os.fspath(pdf_path)
        output_dir = os.fspath(output_dir)

        paper_name = os.path.splitext(os.path.basename(pdf_path))[0]
        paper_dir = paper_output_dir(pdf_path, output_dir)
        os.makedirs(paper_dir, exist_ok=True)

        print(f"Processing {paper_name} with PDF-native extraction...")

        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:
            print(f"Error opening PDF natively: {exc}")
            return False

        full_text_parts = []
        total_images = 0
        pages_dir = os.path.join(paper_dir, "pages")
        os.makedirs(pages_dir, exist_ok=True)

        for page_index in range(doc.page_count):
            human_page_number = page_index + 1
            page = doc.load_page(page_index)
            page_output_path = os.path.join(pages_dir, f"page_{page_index}")
            os.makedirs(page_output_path, exist_ok=True)

            text = page.get_text("text", sort=True).strip()
            result_path = os.path.join(page_output_path, "result.mmd")
            with open(result_path, "w", encoding="utf-8") as file_handle:
                file_handle.write(text)

            page_images_dir = os.path.join(page_output_path, "images")
            os.makedirs(page_images_dir, exist_ok=True)

            page_image_count = 0
            seen_xrefs = set()
            for image_index, image_info in enumerate(page.get_images(full=True), start=1):
                xref = image_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                extracted = doc.extract_image(xref)
                ext = extracted.get("ext", "bin")
                image_path = os.path.join(
                    page_images_dir,
                    f"image_{image_index:02d}_xref_{xref}.{ext}",
                )
                with open(image_path, "wb") as image_handle:
                    image_handle.write(extracted["image"])
                page_image_count += 1
                total_images += 1

            full_text_parts.append(f"## Page {page_index}\n\n{text}")
            print(
                f"Page {human_page_number}/{doc.page_count} success. "
                f"Extracted {len(text)} chars and {page_image_count} embedded images."
            )

        with open(
            os.path.join(paper_dir, "full_extracted.md"),
            "w",
            encoding="utf-8",
        ) as file_handle:
            file_handle.write("\n\n".join(full_text_parts))

        print(
            f"Finished processing {paper_name} with PDF-native extraction. "
            f"Total embedded images: {total_images}"
        )
        return True


class OCREngine:
    method_name = "ocr"

    def __init__(self, model_name=DEFAULT_OCR_MODEL, device="cuda"):
        self.device = device
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.attn_implementation = "eager"
        self._load_model()

    def _preferred_torch_dtype(self):
        if self.device.startswith("cuda") and torch.cuda.is_available():
            return torch.bfloat16
        return torch.float32

    def _load_model(self):
        print(f"Loading {self.model_name}...")

        if self.device.startswith("cuda") and torch.cuda.is_available() and _has_flash_attention_2():
            self.attn_implementation = "flash_attention_2"
        else:
            self.attn_implementation = "eager"
            if self.device.startswith("cuda"):
                print("flash-attn is unavailable. Falling back to eager attention.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_safetensors=True,
            _attn_implementation=self.attn_implementation,
        )
        self.model = self.model.eval().to(self.device).to(self._preferred_torch_dtype())
        print(f"Model loaded successfully with {self.attn_implementation}.")

    def _page_count(self, pdf_path):
        pdf_info = pdfinfo_from_path(pdf_path, timeout=600)
        return int(pdf_info["Pages"])

    def _render_page(self, pdf_path, page_number):
        images = convert_from_path(
            pdf_path,
            first_page=page_number,
            last_page=page_number,
            timeout=600,
        )
        if not images:
            raise RuntimeError(f"Failed to render page {page_number}")
        return images[0]

    def _resolve_ocr_profile(self, pdf_path):
        layout_result = classify_pdf_layout(pdf_path)
        layout_label = layout_result["label"]
        profile_name = layout_label if layout_label in OCR_PROFILES else DEFAULT_OCR_PROFILE
        profile = dict(OCR_PROFILES[profile_name])
        return profile_name, profile, layout_result

    def process_pdf(self, pdf_path, output_dir):
        pdf_path = os.fspath(pdf_path)
        output_dir = os.fspath(output_dir)

        paper_name = os.path.splitext(os.path.basename(pdf_path))[0]
        paper_dir = paper_output_dir(pdf_path, output_dir)
        os.makedirs(paper_dir, exist_ok=True)

        print(f"Processing {paper_name}...")

        try:
            page_count = self._page_count(pdf_path)
        except Exception as exc:
            print(f"Error reading PDF metadata: {exc}")
            return False

        profile_name, ocr_profile, layout_result = self._resolve_ocr_profile(pdf_path)
        print(
            "Detected layout: "
            f"{layout_result['label']} ({layout_result['reason']}). "
            f"Using OCR profile '{profile_name}' with "
            f"base_size={ocr_profile['base_size']}, "
            f"image_size={ocr_profile['image_size']}, "
            f"crop_mode={ocr_profile['crop_mode']}, "
            f"save_results={ocr_profile['save_results']}."
        )

        full_text_parts = []
        pages_dir = os.path.join(paper_dir, "pages")
        os.makedirs(pages_dir, exist_ok=True)
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "

        for page_index in range(page_count):
            human_page_number = page_index + 1
            print(f"Processing page {human_page_number}/{page_count}...")

            try:
                image = self._render_page(pdf_path, human_page_number)
            except Exception as exc:
                print(f"Error rendering page {human_page_number}: {exc}")
                raise

            page_output_path = os.path.join(pages_dir, f"page_{page_index}")
            os.makedirs(page_output_path, exist_ok=True)

            image_path = os.path.join(page_output_path, "original.jpg")
            image.save(image_path)

            try:
                response = self.model.infer(
                    self.tokenizer,
                    prompt=prompt,
                    image_file=image_path,
                    output_path=page_output_path,
                    base_size=ocr_profile["base_size"],
                    image_size=ocr_profile["image_size"],
                    crop_mode=ocr_profile["crop_mode"],
                    save_results=ocr_profile["save_results"],
                )

                result_mmd_path = os.path.join(page_output_path, "result.mmd")
                if not os.path.exists(result_mmd_path):
                    fallback_text = response if isinstance(response, str) else str(response)
                    raise RuntimeError(
                        f"Missing result.mmd for page {page_index}. Model returned: {fallback_text}"
                    )

                with open(result_mmd_path, "r", encoding="utf-8") as file_handle:
                    extracted_text = file_handle.read()

                if not is_valid_ocr_text(extracted_text):
                    fallback_text = response if isinstance(response, str) else str(response)
                    raise RuntimeError(
                        f"Invalid OCR output for page {page_index}. Model returned: {fallback_text}"
                    )

                full_text_parts.append(f"## Page {page_index}\n\n{extracted_text}")
                print(f"Page {human_page_number} success. Extracted {len(extracted_text)} chars.")

            except Exception as exc:
                print(f"Error processing page {human_page_number}: {exc}")
                raise

        with open(
            os.path.join(paper_dir, "full_extracted.md"),
            "w",
            encoding="utf-8",
        ) as file_handle:
            file_handle.write("\n\n".join(full_text_parts))

        print(f"Finished processing {paper_name}")
        return True


def build_extraction_engine(
    method=DEFAULT_EXTRACTION_METHOD,
    model_name=DEFAULT_OCR_MODEL,
    device="cuda",
):
    if method == "native":
        return PDFNativeExtractor()
    if method == "ocr":
        return OCREngine(model_name=model_name, device=device)
    raise ValueError(f"Unsupported extraction method: {method}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("output_dir", help="Directory for OCR output")
    parser.add_argument(
        "--method",
        choices=["native", "ocr"],
        default=DEFAULT_EXTRACTION_METHOD,
        help="Stage 2 extraction backend",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_OCR_MODEL,
        help="OCR model name",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device to use for OCR",
    )
    args = parser.parse_args()

    try:
        engine = build_extraction_engine(
            method=args.method,
            model_name=args.model_name,
            device=args.device,
        )
        success = engine.process_pdf(args.pdf_path, args.output_dir)
        if not success:
            sys.exit(1)
    except Exception as exc:
        print(f"Stage 2 extraction crash: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
