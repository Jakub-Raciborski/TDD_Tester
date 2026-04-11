import os
import json
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ================ CONFIG ================
CSV_PATH = "../Data/API_Data/apps_test_data.csv"
INPUT_COLUMN = "prompt_testcase"
ROW_START_1BASED = 1
MODEL = "gemini-3-flash-preview"
JSONL_INPUT_FILE = "batch.jsonl"
TEMPERATURE = 0.0
MAX_TOKENS = 2000


# ========================================

def prepare_jsonl():
    print(f"Wczytywanie {CSV_PATH} od wiersza {ROW_START_1BASED}...")
    df = pd.read_csv(CSV_PATH)
    df = df.iloc[ROW_START_1BASED - 1:]
    df = df.reset_index(drop=False)

    total_requests = len(df)
    print(f"{total_requests} zapytań do przetworzenia.")

    with open(JSONL_INPUT_FILE, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            row_index = row['index']
            prompt = str(row[INPUT_COLUMN])

            batch_request = {
                "id": f"req-{row_index}",
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": prompt}]
                        }
                    ],
                    "generationConfig": {
                        "temperature": TEMPERATURE,
                        "maxOutputTokens": MAX_TOKENS
                    }
                }
            }
            f.write(json.dumps(batch_request, ensure_ascii=False) + "\n")

    print(f"Plik {JSONL_INPUT_FILE} został pomyślnie utworzony!")
    return total_requests


def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Brak GEMINI_API_KEY w pliku .env")

    # 1. Przygotowanie pliku JSONL
    total = prepare_jsonl()
    if total == 0:
        print("Brak danych do przetworzenia.")
        return

    # 2. Upload pliku do Google (File API)
    client = genai.Client(api_key=api_key)
    print("\nWysyłanie pliku JSONL do serwerów Google...")
    uploaded_file = client.files.upload(
        file=JSONL_INPUT_FILE,
        config=types.UploadFileConfig(mime_type="application/jsonl")
    )
    print(f"Plik wgrany.")

    # 4. Tworzenie Batch Job z wgranym plikiem
    batch_job = client.batches.create(
        model=MODEL,
        src=uploaded_file.name,
        config=types.CreateBatchJobConfig(display_name="APPS_TRAIN")
    )

    print("=" * 60)
    print("ZADANIE WYSŁANE POMYŚLNIE!")
    print(f"ID Zadania: {batch_job.name}")
    print(f"Status: {batch_job.state}")
    print("=" * 60)



if __name__ == "__main__":
    main()