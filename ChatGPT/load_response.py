import os
import json
from openai import OpenAI
from dotenv import load_dotenv

BATCH_METADATA_FILE = "batch_metadata_apps_test.json"
OUTPUT_RESULTS_FILE = "batch_results_apps_test.jsonl"

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Brak OPENAI_API_KEY w .env")

client = OpenAI(api_key=api_key)

with open(BATCH_METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)
batch_id = metadata["batch_id"]

def check_status():
    batch = client.batches.retrieve(batch_id)
    print("Status:", batch.status)
    return batch

def download_results(batch):
    if not getattr(batch, "output_file_id", None):
        print("Brak output_file_id.")
        return
    print("Downloading results...")
    result = client.files.content(batch.output_file_id)
    with open(OUTPUT_RESULTS_FILE, "wb") as f:
        f.write(result.read())
    print("Results saved to:", OUTPUT_RESULTS_FILE)

batch = check_status()
if batch.status == "completed":
    download_results(batch)
elif batch.status in ["failed", "expired"]:
    print("Przetwarzanie batch zakończone z powodu błędu:", batch.errors)
else:
    print("Batch jeszcze się przetwarza.")