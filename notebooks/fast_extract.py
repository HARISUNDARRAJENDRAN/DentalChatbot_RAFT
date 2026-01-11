# ⚡ FAST Parallel PDF Text Extraction
# Run this cell instead of the slow sequential one

import fitz  # PyMuPDF
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.notebook import tqdm
import multiprocessing

# Paths
DRIVE_FOLDER = "/content/drive/MyDrive/MyDentalBooks"
OUTPUT_DIR = "/content/drive/MyDrive/RAFT_dental_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_file = os.path.join(OUTPUT_DIR, "raw_pages.jsonl")

# Get all PDFs
all_pdfs = [f for f in os.listdir(DRIVE_FOLDER) if f.lower().endswith('.pdf')]
print(f"Found {len(all_pdfs)} PDFs to extract")

def extract_single_pdf(pdf_name):
    """Extract text from a single PDF. Returns list of page dicts."""
    pdf_path = os.path.join(DRIVE_FOLDER, pdf_name)
    pages = []

    try:
        doc = fitz.open(pdf_path)

        # Extract category from filename
        match = re.match(r'\[([^\]]+)\]', pdf_name)
        category = match.group(1) if match else "Unknown"

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            text = " ".join(text.split())  # Clean whitespace

            if len(text) >= 100:  # Skip near-empty pages
                pages.append({
                    "text": text,
                    "source": pdf_name,
                    "category": category,
                    "page_number": page_num + 1,
                    "total_pages": len(doc)
                })

        doc.close()
        return pdf_name, pages, None
    except Exception as e:
        return pdf_name, [], str(e)

# Parallel extraction
NUM_WORKERS = min(8, multiprocessing.cpu_count())  # Use 8 workers or CPU count
print(f"Using {NUM_WORKERS} parallel workers")
print(f"Output: {output_file}\n")

total_pages = 0
processed = 0
failed = 0
failed_files = []

with open(output_file, "w", encoding="utf-8") as f:
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks
        futures = {executor.submit(extract_single_pdf, pdf): pdf for pdf in all_pdfs}

        # Process as they complete
        for future in tqdm(as_completed(futures), total=len(all_pdfs), desc="Extracting"):
            pdf_name, pages, error = future.result()

            if error:
                failed += 1
                failed_files.append((pdf_name, error))
            elif pages:
                for page in pages:
                    f.write(json.dumps(page, ensure_ascii=False) + "\n")
                    total_pages += 1
                processed += 1
            else:
                failed += 1

print(f"\n✓ Extraction complete!")
print(f"  Processed: {processed} PDFs")
print(f"  Failed: {failed} PDFs")
print(f"  Total pages: {total_pages}")

if failed_files:
    print(f"\nFailed files (first 5):")
    for name, err in failed_files[:5]:
        print(f"  - {name[:50]}...: {err}")
