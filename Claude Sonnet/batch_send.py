import anthropic
import pandas as pd
from dotenv import load_dotenv
import os

# === KONFIGURACJA ===
CSV_PATH = "../Data/API_Data/apps_test_data.csv"
INPUT_COLUMN    = "prompt_testcase"
MODEL           = "claude-sonnet-4-6"
MAX_TOKENS      = 500
BATCH_ID_FILE   = "batch_id.txt"   # tutaj zostanie zapisany ID batcha

# =====================

load_dotenv()
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

if not CLAUDE_API_KEY:
    raise ValueError("Nie znaleziono CLAUDE_API_KEY w pliku .env")


def main():
    # Wczytaj CSV
    print(f"Wczytuję plik: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    if INPUT_COLUMN not in df.columns:
        raise ValueError(
            f"Kolumna '{INPUT_COLUMN}' nie istnieje w pliku CSV.\n"
            f"Dostępne kolumny: {list(df.columns)}"
        )

    # Zbuduj listę zapytań dla Batch API
    requests = []
    skipped = 0

    for idx, row in df.iterrows():
        prompt = str(row[INPUT_COLUMN]).strip()

        if not prompt or prompt.lower() in ("nan", "none", ""):
            skipped += 1
            continue

        requests.append(
            anthropic.types.message_create_params.MessageCreateParamsNonStreaming(
                custom_id=f"row-{idx}",
                params={
                    "model": MODEL,
                    "max_tokens": MAX_TOKENS,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                },
            )
        )

    if not requests:
        print("Brak zapytań do wysłania (wszystkie wiersze są puste).")
        return

    print(f"Przygotowano {len(requests)} zapytań (pominięto pustych: {skipped}).")

    # Wyślij batch
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    print("Wysyłanie batcha do API...")
    batch = client.messages.batches.create(requests=requests)

    print(f"\nBatch wysłany pomyślnie!")
    print(f"   Batch ID     : {batch.id}")
    print(f"   Status       : {batch.processing_status}")
    print(f"   Wygasa       : {batch.expires_at}")
    print(f"   Liczba zadań : {batch.request_counts.processing}")

    # Zapisz batch_id do pliku
    with open(BATCH_ID_FILE, "w") as f:
        f.write(batch.id)

    print(f"\nBatch ID zapisano do pliku: {BATCH_ID_FILE}")
    print("Uruchom skrypt batch_receive.py, żeby sprawdzić status i odebrać wyniki.")


if __name__ == "__main__":
    main()
