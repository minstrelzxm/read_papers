import subprocess
import time
import sys
import os

def run_ocr():
    cmd = [sys.executable, "src/ocr_engine.py"]
    
    while True:
        print("\n[Robust Runner] Starting OCR Engine...")
        process = subprocess.Popen(cmd)
        exit_code = process.wait()
        
        if exit_code == 0:
            print("[Robust Runner] OCR Engine finished successfully.")
            break
        else:
            print(f"[Robust Runner] OCR Engine crashed with exit code {exit_code}. Restarting in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    # Ensure we are in the project root
    if not os.path.exists("src/ocr_engine.py"):
        print("Error: Run this script from the project root (where src/ is located).")
        sys.exit(1)
        
    run_ocr()
