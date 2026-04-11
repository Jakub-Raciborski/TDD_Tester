"""
SKRYPT 2 — Sprawdzanie statusu i odbieranie wyników z Batch API
================================================================
Odczytuje BATCH_ID z pliku batch_id.txt, sprawdza status batcha.

Jeśli batch jest gotowy (processing_status == "ended"), pobiera
wyniki i zapisuje je do pliku claude_response.csv.

Można uruchamiać wielokrotnie — dopóki batch nie jest gotowy,
skrypt wypisze aktualny postęp i zakończy działanie.

Użycie:
    python batch_receive.py

    # lub z automatycznym odpytywaniem co N minut:
    watch -n 120 python batch_receive.py

Wymagania:
    pip install anthropic pandas python-dotenv
"""

import anthropic
import pandas as pd
from dotenv import load_dotenv
import os

# === KONFIGURACJA ===
CSV_PATH = "../Data/API_Data/apps_test_data.csv"
INPUT_COLUMN     = "prompt_testcase"
OUTPUT_CSV       = "claude_response.csv"
BATCH_ID_FILE    = "batch_id.txt"

# =====================

load_dotenv()
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

if not CLAUDE_API_KEY:
    raise ValueError("Brak CLAUDE_API_KEY w pliku .env")


def main():
    if not os.path.exists(BATCH_ID_FILE):
        raise FileNotFoundError(
            f"Nie znaleziono pliku '{BATCH_ID_FILE}'.\n"
            "Najpierw uruchom skrypt batch_send.py."
        )

    with open(BATCH_ID_FILE, "r") as f:
        batch_id = f.read().strip()

    if not batch_id:
        raise ValueError(f"Plik '{BATCH_ID_FILE}' jest pusty.")

    print(f"Sprawdzam status batcha: {batch_id}")

    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    batch = client.messages.batches.retrieve(batch_id)

    counts = batch.request_counts
    print(f"\nStatus         : {batch.processing_status}")
    print(f"Przetwarzanych : {counts.processing}")
    print(f"Zakończonych   : {counts.succeeded}")
    print(f"Błędów         : {counts.errored}")
    print(f"Anulowanych    : {counts.canceled}")
    print(f"Wygasłych      : {counts.expired}")

    if batch.processing_status != "ended":
        print(
            "\nBatch nie jest jeszcze gotowy. "
        )
        return

    print("\nBatch gotowy. Pobieram wyniki...")


    results: dict[int, str] = {}

    for entry in client.messages.batches.results(batch_id):
        try:
            row_idx = int(entry.custom_id.split("-", 1)[1])
        except (ValueError, IndexError):
            print(f"Nieoczekiwany custom_id: {entry.custom_id} — pomijam.")
            continue

        if entry.result.type == "succeeded":
            text = entry.result.message.content[0].text
            results[row_idx] = text
        elif entry.result.type == "errored":
            error_msg = f"BŁĄD API: {entry.result.error}"
            results[row_idx] = error_msg
            print(f"Wiersz {row_idx}: {error_msg}")
        elif entry.result.type == "expired":
            results[row_idx] = "BŁĄD: Zapytanie wygasło przed przetworzeniem"
        elif entry.result.type == "canceled":
            results[row_idx] = "BŁĄD: Zapytanie zostało anulowane"

    print(f"Odebrano odpowiedzi dla {len(results)} wierszy.")

    # Wczytaj oryginalny CSV i dodaj kolumnę z odpowiedziami
    df = pd.read_csv(CSV_PATH)

    df["Claude Sonnet 4.6 asserts"] = df.index.map(
        lambda idx: results.get(idx, "")
    )

    # Zapisz wynik
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWyniki zapisane do: {OUTPUT_CSV}")

    # Podsumowanie
    filled = df["Claude Sonnet 4.6 asserts"].astype(str).str.strip()
    filled = filled[filled != ""]
    print(f"Wypełnionych wierszy : {len(filled)} / {len(df)}")


if __name__ == "__main__":
    main()
