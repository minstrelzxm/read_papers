import os
import torch
import sys
import logging
from transformers import AutoModel, AutoTokenizer, logging as hf_logging

# Suppress warnings
hf_logging.set_verbosity_error()

from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm
import numpy as np

class OCREngine:
    def __init__(self, model_name='deepseek-ai/DeepSeek-OCR', device='cuda'):
        self.device = device
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        print(f"Loading {self.model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            # Use default attention (likely sdpa or eager) to avoid flash-attn dependency issues
            # and device-side asserts with incompatible kernels/torch versions.
            self.model = AutoModel.from_pretrained(
                self.model_name, 
                trust_remote_code=True, 
                use_safetensors=True,
                attn_implementation="eager" 
            )
            
            self.model = self.model.eval().to(self.device)
            if self.device == 'cuda':
                # Force float32 for stability if lower precision fails
                pass 
                # self.model = self.model.to(torch.float16)
                
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise e

    def process_pdf(self, pdf_path, output_dir):
        paper_name = os.path.splitext(os.path.basename(pdf_path))[0]
        paper_output_dir = os.path.join(output_dir, paper_name)
        os.makedirs(paper_output_dir, exist_ok=True)
        
        print(f"Processing {paper_name}...")
        
        # Convert PDF to images
        try:
            # Set a meaningful timeout (e.g., 60 seconds per PDF) to avoid hangs
            images = convert_from_path(pdf_path, timeout=60)
        except Exception as e:
            print(f"Error converting PDF to images: {e}")
            return False

        full_text = ""
        
        # Create a specific directory for raw page outputs to avoid clutter
        pages_dir = os.path.join(paper_output_dir, "pages")
        os.makedirs(pages_dir, exist_ok=True)
        
        for i, image in enumerate(images):
            # if i > 0: break # Debug removed
            
            print(f"Processing Page {i}, Size: {image.size}, Mode: {image.mode}")

            page_output_path = os.path.join(pages_dir, f"page_{i}")
            os.makedirs(page_output_path, exist_ok=True)
            
            # Save original page image
            image_path = os.path.join(page_output_path, "original.jpg")
            image.save(image_path)
            
            # Run OCR
            prompt = "<image>\n<|grounding|>Convert the document to markdown. "
            
            try:
                res = self.model.infer(
                    self.tokenizer,
                    prompt=prompt,
                    image_file=image_path,
                    output_path=page_output_path,
                    base_size=1024,
                    image_size=1024, # Using base size from example for better quality
                    crop_mode=False,
                    save_results=True,
                    test_compress=False
                )
                
                # DeepSeek-OCR saves the result to 'result.mmd' in the output path when save_results=True.
                # The return value 'res' might be None or empty.
                result_mmd_path = os.path.join(page_output_path, "result.mmd")
                
                extracted_text = ""
                if os.path.exists(result_mmd_path):
                    try:
                        with open(result_mmd_path, "r", encoding="utf-8") as f:
                            extracted_text = f.read()
                    except Exception as e:
                         print(f"Error reading result.mmd for page {i}: {e}")
                else:
                    # Fallback if return value allows or file missing
                    extracted_text = res if isinstance(res, str) else str(res)

                full_text += f"\n\n## Page {i}\n\n{extracted_text}"
                print(f"Page {i} success. Extracted {len(extracted_text)} chars.")
                
            except Exception as e:
                print(f"Error processing page {i}: {e}")
                # Re-raise to crash the subprocess so main.py knows to move on (and clear CUDA context)
                raise e

        # Save full extracted text
        with open(os.path.join(paper_output_dir, "full_extracted.md"), "w", encoding="utf-8") as f:
            f.write(full_text)
            
        print(f"Finished processing {paper_name}")
        return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", nargs="?", help="Path to PDF")
    parser.add_argument("output_dir", nargs="?", help="Output directory")
    args = parser.parse_args()

    # If arguments are provided, run single paper mode (subprocess mode)
    if args.pdf_path and args.output_dir:
        try:
            engine = OCREngine()
            success = engine.process_pdf(args.pdf_path, args.output_dir)
            if not success:
                sys.exit(1)
        except Exception as e:
            print(f"OCR Crash: {e}", file=sys.stderr)
            sys.exit(1)
            
    # Otherwise run batch mode (default/debug behavior)
    else:
        # Test on files in directory
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        input_dir = os.path.join(base_dir, 'original_papers')
        output_dir = os.path.join(base_dir, 'extracted_papers')
        
        if not os.path.exists(input_dir):
            print("No arguments provided and 'original_papers' not found.")
            sys.exit(1)

        files = sorted([f for f in os.listdir(input_dir) if f.endswith('.pdf')])
        
        if files:
            print(f"Found {len(files)} PDFs. Starting batch OCR...")
            engine = OCREngine()
            
            for i, pdf_file in enumerate(tqdm(files)):
                paper_name = os.path.splitext(pdf_file)[0]
                existing_output = os.path.join(output_dir, paper_name)
                
                # Skip if already processed
                if os.path.exists(existing_output) and os.path.exists(os.path.join(existing_output, "full_extracted.md")):
                    continue
                    
                print(f"[{i+1}/{len(files)}] Processing {paper_name}...")
                try:
                    engine.process_pdf(os.path.join(input_dir, pdf_file), output_dir)
                except Exception as e:
                    print(f"Failed to process {paper_name}: {e}")
                    continue
        else:
            print("No PDF files found to process.")
