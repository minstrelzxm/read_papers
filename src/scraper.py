import openreview
import os
import requests
from tqdm import tqdm
import concurrent.futures

def get_neurips_2025_papers():
    print("Connecting to OpenReview...")
    client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
    
    # NeurIPS 2025 Conference ID
    venue_id = 'NeurIPS.cc/2025/Conference'
    
    # Fetching accepted papers
    # Usually accepted papers are under the submission invitation but have a decision note, 
    # or they are published under the conference id directly.
    # For accepted papers, better to look for the 'neurips.cc/2025/Conference/-/Submission' inv
    # and filter for accepted decisions if possible, OR look for papers that are in the venue.
    
    print("Fetching submissions...")
    # Trying to fetch all submissions accepted to the conference
    # Note: 'content.venueid' might differ slightly, but usually it is the venue_id
    
    # Strategy: Get all accepted submissions.
    # OpenReview API v2 usually uses 'content.venueid' to indicate acceptance venue.
    submissions = client.get_all_notes(content={'venueid': venue_id})
    
    print(f"Found {len(submissions)} papers associated with {venue_id}")
    return submissions

import random
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def get_session():
    session = requests.Session()
    retry = Retry(
        total=5, 
        read=5, 
        connect=5, 
        backoff_factor=1,  # 1s, 2s, 4s, 8s, 16s...
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

from requests.exceptions import ChunkedEncodingError, ConnectionError
from urllib3.exceptions import ProtocolError, IncompleteRead

def download_pdf(paper, output_dir):
    # Random sleep to prevent synchronized hammering
    time.sleep(random.uniform(0.5, 2.0))
    
    try:
        content = paper.content
        title = content.get('title', {}).get('value', 'Untitled')
        
        pdf_url = None
        if 'pdf' in content and 'value' in content['pdf']:
             pdf_url = f"https://openreview.net{content['pdf']['value']}"
        elif 'file' in content and 'value' in content['file']: 
             pdf_url = f"https://openreview.net{content['file']['value']}"
        
        if not pdf_url:
            pdf_url = f"https://openreview.net/pdf?id={paper.id}"

        # Sanitize title
        safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ']).rstrip().replace(" ", "_")
        filename = f"{safe_title}_{paper.id}.pdf"
        filepath = os.path.join(output_dir, filename)
        
        if os.path.exists(filepath):
            # Check for valid PDF size (sometimes empty files occur on fail)
            if os.path.getsize(filepath) > 1024: 
                return {"status": "skipped", "file": filename, "msg": "Exists"}
            else:
                os.remove(filepath) 

        session = get_session()
        try:
            response = session.get(pdf_url, stream=True, timeout=60)
            
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk: 
                            f.write(chunk)
                return {"status": "success", "file": filename, "msg": "Downloaded"}
            else:
                return {"status": "error", "file": filename, "msg": f"HTTP {response.status_code}", "title": title}
                
        except (ChunkedEncodingError, ProtocolError, IncompleteRead, ConnectionError) as e:
            # Network cut out mid-download
            if os.path.exists(filepath):
                os.remove(filepath) # Delete partial file
            return {"status": "error", "file": filename, "msg": f"Network Error: {str(e)}", "title": title}
            
    except Exception as e:
        return {"status": "error", "file": paper.id, "msg": str(e), "title": title if 'title' in locals() else 'Unknown'}

def process_downloads(papers, output_dir, max_retries=3):
    total = len(papers)
    failures = []
    
    print(f"Processing {total} papers...")
    
    # 5 workers is safer for OpenReview to avoid 429
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(download_pdf, paper, output_dir): paper for paper in papers}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=total):
            result = future.result()
            if result['status'] == 'error':
                print(f"\n[ERROR] Failed: {result.get('title', 'Unknown')} - {result['msg']}")
                failures.append(futures[future]) # Store paper object for retry
            elif result['status'] == 'success':
                pass # success
            # skipped is silent
            
    return failures

def main():
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../original_papers'))
    os.makedirs(output_dir, exist_ok=True)
    
    papers = get_neurips_2025_papers()
    
    if not papers:
        print("No papers found. Please check the venue ID or API access.")
        return

    print(f"Target Directory: {output_dir}")
    
    # Retry Loop
    current_batch = papers
    retry_count = 0
    max_retries = 3
    
    while current_batch and retry_count < max_retries:
        if retry_count > 0:
            print(f"\n--- Retry Attempt {retry_count}/{max_retries} for {len(current_batch)} failed papers ---")
        
        failed_papers = process_downloads(current_batch, output_dir)
        
        if not failed_papers:
            print("\nAll papers processed successfully!")
            break
            
        current_batch = failed_papers
        retry_count += 1
        
    if current_batch:
        print(f"\nWarning: {len(current_batch)} papers failed after all retries.")
        with open("failed_downloads.txt", "w") as f:
            for p in current_batch:
                title = p.content.get('title', {}).get('value', 'Untitled')
                f.write(f"{p.id} - {title}\n")
        print("Failed list saved to failed_downloads.txt")

if __name__ == "__main__":
    main()
