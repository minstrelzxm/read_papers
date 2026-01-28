import os
import sys
import subprocess
import argparse
from src.scraper import get_neurips_2025_papers, download_pdf
from src.ocr_engine import OCREngine
from src.analyzer import OpenAIAnalyzer, LocalVLMAnalyzer, PaperContent
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Automated Paper Reading Pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of papers to process")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading papers")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR processing")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis")
    
    # Analyzer Args
    parser.add_argument("--provider", choices=["local", "online"], default="local", help="Analyzer provider")
    parser.add_argument("--model", default=None, help="Model name for analysis")
    parser.add_argument("--api_key", default=None, help="OpenAI API Key for online provider")
    
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    original_dir = os.path.join(base_dir, "original_papers")
    extracted_dir = os.path.join(base_dir, "extracted_papers")
    
    # 1. Scrape & Download
    if not args.skip_download:
        print("--- Step 1: Downloading Papers ---")
        os.makedirs(original_dir, exist_ok=True)
        papers = get_neurips_2025_papers()
        
        if args.limit:
            papers = papers[:args.limit]
            
        import concurrent.futures
        
        print(f"Downloading {len(papers)} papers...")
        # Use parallel downloads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(download_pdf, paper, original_dir) for paper in papers]
            # Use separate tqdm loop to monitor completion
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(papers)):
                pass
    
    # 2. Process Papers
    pdf_files = [f for f in os.listdir(original_dir) if f.endswith('.pdf')]
    if args.limit:
        pdf_files = pdf_files[:args.limit]
        
    if not pdf_files:
        print("No papers found to process.")
        return

    # Initialize Models only if needed
    ocr_engine = None
    analyzer = None
    
    if not args.skip_ocr:
        print("--- Step 2: OCR Processing ---")
        # Optimization: Do NOT load OCREngine here. 
        # We run it in isolated subprocesses to manage VRAM and stability.
        pass

    if not args.skip_analysis:
        print("--- Step 3: Analysis ---")
        try:
            if args.provider == "online":
                model_name = args.model if args.model else "gpt-5"
                print(f"Initializing Online Analyzer ({model_name})...")
                analyzer = OpenAIAnalyzer(model_name=model_name, api_key=args.api_key)
            else:
                model_name = args.model if args.model else "Qwen/Qwen3-VL-4B-Thinking"
                print(f"Initializing Local VLM Analyzer ({model_name})...")
                analyzer = LocalVLMAnalyzer(model_name=model_name)
                
        except Exception as e:
             print(f"Failed to initialize Analyzer: {e}")
             return

    for pdf_file in tqdm(pdf_files, desc="Processing Pipeline"):
        pdf_path = os.path.join(original_dir, pdf_file)
        paper_name = os.path.splitext(pdf_file)[0]
        paper_extracted_path = os.path.join(extracted_dir, paper_name)
        
        # OCR
        if not args.skip_ocr:
            full_extracted_path = os.path.join(paper_extracted_path, "full_extracted.md")
            
            # Check if already extracted (exists and not empty)
            already_extracted = False
            if os.path.exists(full_extracted_path):
                if os.path.getsize(full_extracted_path) > 0:
                    already_extracted = True
            
            if not already_extracted:
                print(f"Running OCR on {paper_name}...")
                # Run OCR in a separate process to isolate CUDA crashes
                try:
                    cmd = [sys.executable, "src/ocr_engine.py", pdf_path, extracted_dir]
                    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', timeout=1200) # 20 min timeout per paper
                    
                    if result.returncode != 0:
                        print(f"OCR Failed for {paper_name}")
                        print(f"  Stdout: {result.stdout}")
                        print(f"  Stderr: {result.stderr}")
                        
                        # Cleanup corrupted output
                        if os.path.exists(paper_extracted_path):
                            import shutil
                            shutil.rmtree(paper_extracted_path)
                            print(f"  Cleaned up partial output for {paper_name}")
                    else:
                         pass # Success
                         
                except subprocess.TimeoutExpired:
                     print(f"OCR Timed out for {paper_name}")
                     if os.path.exists(paper_extracted_path):
                        import shutil
                        shutil.rmtree(paper_extracted_path)
                except Exception as e:
                    print(f"Error launching OCR subprocess for {paper_name}: {e}")
                    if os.path.exists(paper_extracted_path):
                        import shutil
                        shutil.rmtree(paper_extracted_path)
            else:
                pass
        
        # Analysis
        if not args.skip_analysis and analyzer:
            # For analysis, we check if report exists
            if os.path.exists(os.path.join(paper_extracted_path, "full_extracted.md")):
                if not os.path.exists(os.path.join(paper_extracted_path, "analysis_report.md")):
                     print(f"Analyzing {paper_name}...")
                     # Load content using the new class
                     content = PaperContent(paper_extracted_path)
                     report = analyzer.analyze(content)
                     
                     if report:
                         with open(os.path.join(paper_extracted_path, "analysis_report.md"), "w") as f:
                             f.write(report)
                         print(f"Report saved for {paper_name}")
                else:
                     pass
            else:
                print(f"Cannot analyze {paper_name}: No extracted text.")

    print("Pipeline Completed.")

if __name__ == "__main__":
    main()
