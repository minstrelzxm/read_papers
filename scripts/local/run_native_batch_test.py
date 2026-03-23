import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR_STR = str(ROOT_DIR)
if ROOT_DIR_STR not in sys.path:
    sys.path.insert(0, ROOT_DIR_STR)

from src.analyzer import DEFAULT_LOCAL_MODEL, analyze_paper_folder, build_analyzer
from src.ocr_engine import build_extraction_engine, paper_output_dir


DEFAULT_BATCH_SIZE = 10
DEFAULT_OUTPUT_ROOT = Path("test_folder/native_batch_test")


def select_random_pdfs(original_dir, count, seed=None):
    pdfs = sorted(original_dir.glob("*.pdf"))
    if len(pdfs) < count:
        raise ValueError(f"Requested {count} PDFs but only found {len(pdfs)} in {original_dir}")

    rng = random.Random(seed)
    selected = rng.sample(pdfs, count)
    selected.sort(key=lambda path: path.name.lower())
    return selected


def run_batch(
    pdf_paths,
    output_root,
    provider,
    model_name,
    api_key,
    base_url,
    extraction_method,
    ocr_model,
    ocr_device,
):
    output_root.mkdir(parents=True, exist_ok=True)
    extracted_root = output_root / "extracted_papers"
    extracted_root.mkdir(exist_ok=True)

    analyzer = build_analyzer(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
    )
    extraction_engine = build_extraction_engine(
        method=extraction_method,
        model_name=ocr_model,
        device=ocr_device,
    )

    batch_started = time.perf_counter()
    results = []

    for index, pdf_path in enumerate(pdf_paths, start=1):
        paper_name = pdf_path.stem
        paper_dir = Path(paper_output_dir(pdf_path, extracted_root))
        record = {
            "paper_name": paper_name,
            "pdf_path": str(pdf_path),
            "paper_dir": str(paper_dir),
            "extraction_method": extraction_method,
            "analysis_provider": provider,
            "analysis_model": model_name,
            "extraction_status": "pending",
            "analysis_status": "pending",
        }

        print(f"[{index}/{len(pdf_paths)}] Stage 2 on {paper_name}")
        stage2_started = time.perf_counter()
        try:
            extraction_ok = bool(extraction_engine.process_pdf(pdf_path, extracted_root))
        except Exception as exc:
            extraction_ok = False
            record["extraction_error"] = str(exc)
        record["stage2_seconds"] = round(time.perf_counter() - stage2_started, 2)
        record["extraction_status"] = "passed" if extraction_ok else "failed"

        if extraction_ok:
            print(f"[{index}/{len(pdf_paths)}] Stage 3 on {paper_name}")
            stage3_started = time.perf_counter()
            try:
                report_path = analyze_paper_folder(paper_dir, analyzer)
                record["analysis_report"] = str(report_path)
                record["analysis_status"] = "passed"
            except Exception as exc:
                record["analysis_status"] = "failed"
                record["analysis_error"] = str(exc)
            record["stage3_seconds"] = round(time.perf_counter() - stage3_started, 2)
        else:
            record["analysis_status"] = "skipped"
            record["stage3_seconds"] = 0.0

        results.append(record)
        (output_root / "batch_results.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    batch_seconds = round(time.perf_counter() - batch_started, 2)
    return results, batch_seconds


def write_summary(output_root, selected_pdfs, results, seed, batch_seconds):
    selected_names = [path.name for path in selected_pdfs]
    passed_stage2 = sum(1 for item in results if item["extraction_status"] == "passed")
    passed_stage3 = sum(1 for item in results if item["analysis_status"] == "passed")
    failed_stage2 = [item["paper_name"] for item in results if item["extraction_status"] != "passed"]
    failed_stage3 = [item["paper_name"] for item in results if item["analysis_status"] == "failed"]

    manifest = {
        "seed": seed,
        "selected_pdfs": selected_names,
        "batch_seconds": batch_seconds,
        "stage2_passed": passed_stage2,
        "stage3_passed": passed_stage3,
        "stage2_failed": failed_stage2,
        "stage3_failed": failed_stage3,
    }

    (output_root / "selected_pdfs.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_root / "selected_pdfs.txt").write_text(
        "\n".join(selected_names) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Native Batch Test",
        "",
        f"- Seed: `{seed}`",
        f"- Selected PDFs: {len(selected_pdfs)}",
        f"- Total duration: {batch_seconds:.2f}s",
        f"- Stage 2 passed: {passed_stage2}/{len(results)}",
        f"- Stage 3 passed: {passed_stage3}/{len(results)}",
        "",
        "## Selected PDFs",
        "",
    ]
    lines.extend(f"- `{name}`" for name in selected_names)
    lines.extend(
        [
            "",
            "## Per-paper Results",
            "",
            "| Paper | Stage 2 | Stage 2 s | Stage 3 | Stage 3 s |",
            "| --- | --- | ---: | --- | ---: |",
        ]
    )
    for item in results:
        lines.append(
            "| "
            f"{item['paper_name']} | "
            f"{item['extraction_status']} | "
            f"{item.get('stage2_seconds', 0.0)} | "
            f"{item['analysis_status']} | "
            f"{item.get('stage3_seconds', 0.0)} |"
        )

    if failed_stage2:
        lines.extend(["", "## Stage 2 Failures", ""])
        lines.extend(f"- `{name}`" for name in failed_stage2)

    if failed_stage3:
        lines.extend(["", "## Stage 3 Failures", ""])
        lines.extend(f"- `{name}`" for name in failed_stage3)

    (output_root / "batch_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Run a random batch test using native PDF extraction followed by LLM analysis."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"How many random PDFs to test. Default: {DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling. Defaults to current timestamp-derived randomness.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Directory to store selected-pdf manifests and extracted outputs. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--original-dir",
        type=Path,
        default=Path("original_papers"),
        help="Directory containing source PDFs.",
    )
    parser.add_argument(
        "--provider",
        choices=["local", "local-openai", "openai", "claude", "online"],
        default="local",
        help="Analyzer provider for Stage 3.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LOCAL_MODEL,
        help=f"Analyzer model name. Default: {DEFAULT_LOCAL_MODEL}",
    )
    parser.add_argument("--api_key", default=None, help="API key for online providers.")
    parser.add_argument("--base_url", default=None, help="Base URL for OpenAI-compatible providers.")
    parser.add_argument(
        "--extraction-method",
        choices=["native", "ocr"],
        default="native",
        help="Stage 2 extraction backend. Default: native",
    )
    parser.add_argument(
        "--ocr-model",
        default="deepseek-ai/DeepSeek-OCR-2",
        help="OCR model name if --extraction-method ocr is used.",
    )
    parser.add_argument(
        "--ocr-device",
        default="cuda",
        help="Torch device for OCR if --extraction-method ocr is used.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the output directory before starting the batch.",
    )
    args = parser.parse_args()

    if args.clean and args.output_root.exists():
        shutil.rmtree(args.output_root)

    if args.seed is None:
        args.seed = int(time.time())

    selected_pdfs = select_random_pdfs(args.original_dir, args.count, seed=args.seed)
    results, batch_seconds = run_batch(
        pdf_paths=selected_pdfs,
        output_root=args.output_root,
        provider=args.provider,
        model_name=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        extraction_method=args.extraction_method,
        ocr_model=args.ocr_model,
        ocr_device=args.ocr_device,
    )
    write_summary(
        output_root=args.output_root,
        selected_pdfs=selected_pdfs,
        results=results,
        seed=args.seed,
        batch_seconds=batch_seconds,
    )

    print(f"Batch results saved to {args.output_root}")


if __name__ == "__main__":
    main()
