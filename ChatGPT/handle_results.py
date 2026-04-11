import json
import pandas as pd

# ================= CONFIG =================
CSV_PATH = "../Data/API_Data/apps_test_data.csv"
BATCH_RESULTS_FILE = "batch_results_apps_test.jsonl"
OUTPUT_COLUMN = "ChatGPT-5.4 asserts"
ROW_START_1BASED = 1
# ==========================================

# --- Wczytaj CSV ---
df = pd.read_csv(CSV_PATH)
if OUTPUT_COLUMN not in df.columns:
    df[OUTPUT_COLUMN] = ""
results = []

with open(BATCH_RESULTS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if obj.get("response", {}).get("status_code") != 200:
            results.append("")
            continue

        body = obj["response"]["body"]
        try:
            text = body["output"][0]["content"][0]["text"]
        except (KeyError, IndexError):
            text = ""

        results.append(text)

print(f"Wczytano {len(results)} wyniki.")

# --- Zapis do CSV ---
start_index = ROW_START_1BASED - 1

for i, result in enumerate(results):
    df.loc[start_index + i, OUTPUT_COLUMN] = result

# --- Zapis ---
df.to_csv(CSV_PATH, index=False, encoding="utf-8")

print(f"Zapisano wyniki do {CSV_PATH}")