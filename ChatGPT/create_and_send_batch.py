# create_and_send_batch.py
import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# ================ CONFIG ================
CSV_PATH = "../Data/API_Data/apps_test_data.csv"
INPUT_COLUMN = "prompt_testcase"
ROW_START_1BASED = 1
MODEL = "gpt-5.4"
MAX_TOKENS = 500
TEMPERATURE = 0
BATCH_INPUT_FILE = "batch_input_apps_test.jsonl"
BATCH_METADATA_FILE = "batch_metadata_apps_test.json"
# =========================================

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Brak OPENAI_API_KEY w pliku .env")

client = OpenAI(api_key=api_key)

df = pd.read_csv(CSV_PATH)
df = df.iloc[ROW_START_1BASED - 1:]
df = df.reset_index(drop=False)  # zachowujemy oryginalny indeks

with open(BATCH_INPUT_FILE, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        task = {
            "custom_id": f"req-{row['index']}",
            "method": "POST",
            "url": "/v1/responses",               # <<--- ważne
            "body": {
                "model": MODEL,                   # <<--- gpt-5.2
                "input": str(row[INPUT_COLUMN]),  # prosty string prompt
                "max_output_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE
            }
        }
        f.write(json.dumps(task, ensure_ascii=False) + "\n")

# upload file
batch_file = client.files.create(
    file=open(BATCH_INPUT_FILE, "rb"),
    purpose="batch"
)
print("Input file uploaded. File ID:", batch_file.id)

# create batch job (endpoint wskazany powyżej w body.url, tutaj argument endpoint to general job target)
batch_job = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/responses",
    completion_window="24h"
)
print("Batch ID:", batch_job.id)

metadata = {
    "batch_id": batch_job.id,
    "input_file_id": batch_file.id,
    "model": MODEL,
    "total_prompts": len(df)
}
with open(BATCH_METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("Batch metadata saved.")