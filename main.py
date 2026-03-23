import argparse

from src.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Automated Paper Reading Pipeline")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of papers to process",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading papers",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip OCR processing",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip analysis",
    )
    parser.add_argument(
        "--provider",
        choices=["local", "local-openai", "openai", "claude", "online"],
        default="local",
        help="Analyzer provider",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for analysis",
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="API key for online providers",
    )
    parser.add_argument(
        "--base_url",
        default=None,
        help="Base URL for OpenAI-compatible providers",
    )

    args = parser.parse_args()

    run_pipeline(
        limit=args.limit,
        skip_download=args.skip_download,
        skip_ocr=args.skip_ocr,
        skip_analysis=args.skip_analysis,
        provider=args.provider,
        model_name=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )


if __name__ == "__main__":
    main()
