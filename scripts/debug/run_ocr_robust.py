import subprocess
import sys
import time
from pathlib import Path


def run_ocr():
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        str(repo_root / "main.py"),
        "--skip-download",
        "--skip-analysis",
    ]

    while True:
        print("\n[Robust Runner] Starting OCR pipeline...")
        process = subprocess.Popen(cmd, cwd=repo_root)
        exit_code = process.wait()

        if exit_code == 0:
            print("[Robust Runner] OCR pipeline finished successfully.")
            break

        print(
            "[Robust Runner] OCR pipeline crashed with "
            f"exit code {exit_code}. Restarting in 5 seconds..."
        )
        time.sleep(5)


if __name__ == "__main__":
    run_ocr()
