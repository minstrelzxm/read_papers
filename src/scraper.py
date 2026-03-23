import concurrent.futures
import os
import random
import time

import openreview
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ChunkedEncodingError, ConnectionError
from tqdm import tqdm
from urllib3.exceptions import IncompleteRead, ProtocolError
from urllib3.util.retry import Retry


def get_neurips_2025_papers():
    print("Connecting to OpenReview...")
    client = openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net")

    venue_id = "NeurIPS.cc/2025/Conference"

    print("Fetching submissions...")
    submissions = client.get_all_notes(content={"venueid": venue_id})

    print(f"Found {len(submissions)} papers associated with {venue_id}")
    return submissions


def get_session():
    session = requests.Session()
    retry = Retry(
        total=5,
        read=5,
        connect=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def download_pdf(paper, output_dir):
    output_dir = os.fspath(output_dir)

    time.sleep(random.uniform(0.5, 2.0))

    try:
        content = paper.content
        title = content.get("title", {}).get("value", "Untitled")

        pdf_url = None
        if "pdf" in content and "value" in content["pdf"]:
            pdf_url = f"https://openreview.net{content['pdf']['value']}"
        elif "file" in content and "value" in content["file"]:
            pdf_url = f"https://openreview.net{content['file']['value']}"

        if not pdf_url:
            pdf_url = f"https://openreview.net/pdf?id={paper.id}"

        safe_title = "".join(
            [char for char in title if char.isalpha() or char.isdigit() or char == " "]
        ).rstrip().replace(" ", "_")
        filename = f"{safe_title}_{paper.id}.pdf"
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            if os.path.getsize(filepath) > 1024:
                return {"status": "skipped", "file": filename, "msg": "Exists"}
            os.remove(filepath)

        session = get_session()
        try:
            response = session.get(pdf_url, stream=True, timeout=60)

            if response.status_code == 200:
                with open(filepath, "wb") as file_handle:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file_handle.write(chunk)
                return {"status": "success", "file": filename, "msg": "Downloaded"}

            return {
                "status": "error",
                "file": filename,
                "msg": f"HTTP {response.status_code}",
                "title": title,
            }

        except (ChunkedEncodingError, ProtocolError, IncompleteRead, ConnectionError) as exc:
            if os.path.exists(filepath):
                os.remove(filepath)
            return {
                "status": "error",
                "file": filename,
                "msg": f"Network Error: {exc}",
                "title": title,
            }

    except Exception as exc:
        return {
            "status": "error",
            "file": paper.id,
            "msg": str(exc),
            "title": title if "title" in locals() else "Unknown",
        }


def process_downloads(papers, output_dir, max_workers=5):
    failures = []

    print(f"Processing {len(papers)} papers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_pdf, paper, output_dir): paper for paper in papers
        }

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(papers)):
            result = future.result()
            if result["status"] == "error":
                print(f"\n[ERROR] Failed: {result.get('title', 'Unknown')} - {result['msg']}")
                failures.append(futures[future])

    return failures


def download_papers(
    papers,
    output_dir,
    max_retries=3,
    max_workers=5,
    failed_output_path="failed_downloads.txt",
):
    output_dir = os.fspath(output_dir)
    failed_output_path = os.fspath(failed_output_path)

    os.makedirs(output_dir, exist_ok=True)

    current_batch = list(papers)
    retry_count = 0

    while current_batch and retry_count < max_retries:
        if retry_count > 0:
            print(
                f"\n--- Retry Attempt {retry_count}/{max_retries} "
                f"for {len(current_batch)} failed papers ---"
            )

        failed_papers = process_downloads(
            current_batch,
            output_dir,
            max_workers=max_workers,
        )

        if not failed_papers:
            print("\nAll papers processed successfully!")
            if os.path.exists(failed_output_path):
                os.remove(failed_output_path)
            return []

        current_batch = failed_papers
        retry_count += 1

    if current_batch:
        print(f"\nWarning: {len(current_batch)} papers failed after all retries.")
        with open(failed_output_path, "w", encoding="utf-8") as file_handle:
            for paper in current_batch:
                title = paper.content.get("title", {}).get("value", "Untitled")
                file_handle.write(f"{paper.id} - {title}\n")
        print(f"Failed list saved to {failed_output_path}")

    return current_batch
