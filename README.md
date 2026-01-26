# NeurIPS 2025 Paper Reader & Analyzer

An automated pipeline to scrape, download, OCR, and analyze NeurIPS 2025 papers using advanced VLMs (DeepSeek-OCR) and AI Agents (GPT-5 or Local Qwen-VL).

## Features

- **Automated Scraping**: Fetches all accepted NeurIPS 2025 papers directly from OpenReview.
- **Robust Downloading**: Parallel download handling with automatic retries for failed downloads.
- **Advanced OCR**: Utilizes **DeepSeek-OCR** (running locally via Transformers) to convert PDF papers into structured Markdown.
  - Handles mixed-modality (text + figures + tables).
  - Robust to single/multi-column layouts.
- **AI Analysis**:
  - **Online Mode**: Uses OpenAI's **GPT-5** (requires API Key) to read the extracted text and images and generate a structured research report.
  - **Local Mode**: Uses **Qwen-3-VL-Thinking** models to analyze papers completely offline.
- **Pipeline Architecture**: 
  - Isolated subprocesses for OCR to prevent VRAM fragmentation and ensure stability.
  - Checkpoint system: Skips already downloaded/extracted papers.

## Installation

### Prerequisites
- Linux with NVIDIA GPU (CUDA support required for OCR and Local Analysis).
- Conda installed.
- `poppler-utils` installed (for PDF rasterization).

```bash
sudo apt-get install poppler-utils
```

### Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd read_papers
   ```

2. **Create Conda Environment**:
   ```bash
   conda create -n read_papers python=3.10
   conda activate read_papers
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   *Note: DeepSeek-OCR requires specific `transformers` versions due to attention mechanism changes. If you encounter issues, ensure you are using:*
   ```bash
   pip install transformers==4.46.3
   pip install torch==2.5.1 torchvision
   ```

## Usage

### 1. Run the Full Pipeline
This command will scrape OpenReview, download new papers, run OCR, and analyze them.
```bash
python main.py
```

### 2. Available Options

| Flag | Description |
|------|-------------|
| `--limit <N>` | Process only the first N papers. |
| `--skip-download` | Skip the scraping/downloading step (process existing files only). |
| `--skip-ocr` | Skip the OCR step. |
| `--skip-analysis` | Skip the analysis step. |
| `--provider <online|local>` | Choose analysis provider. Default: `local`. |
| `--model <name>` | Specify model name (e.g., `gpt-5` or `Qwen/Qwen3-VL...`). |
| `--api_key <key>` | API Key for online provider (optional if set in env). |

### Examples

**Process only existing papers locally (No Download):**
```bash
python main.py --skip-download
```

**Use GPT-5 for Analysis:**
```bash
python main.py --skip-download --provider online --model gpt-5 --api_key "sk-..."
```

**Run everything but limit to 5 papers for testing:**
```bash
python main.py --limit 5
```

## Directory Structure

```
├── .conda/                 # Conda environment (ignored by git)
├── original_papers/        # Raw downloaded PDFs (ignored by git)
├── extracted_papers/       # Output directory (ignored by git)
│   └── [Paper_Title]/
│       ├── full_extracted.md    # Full OCR output
│       ├── analysis_report.md   # AI Agent Report
│       └── pages/               # Per-page images and raw data
├── src/
│   ├── analyzer.py         # Logic for OpenAI/Local VLM analysis
│   ├── ocr_engine.py       # DeepSeek-OCR wrapping logic
│   ├── scraper.py          # OpenReview scraping logic
│   └── single_paper_ocr.py # Subprocess entry point for OCR
└── main.py                 # Entry point CLI
```

## Troubleshooting

- **OCR Crash due to encoding**: Ensure your environment supports UTF-8. The code explicitly sets utf-8 encoding for file writes.
- **CUDA Errors**: DeepSeek-OCR can be sensitive to `transformers` versions. Downgrade to `4.46.3` if you see `LlamaFlashAttention2` errors.
- **Timeout**: PDF rasterization has a timeout of 300s. Large papers (30+ pages) might take time.

## License

[Your License Here]
