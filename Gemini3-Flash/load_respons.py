import os
import json
import pandas as pd
from dotenv import load_dotenv
from google import genai

# ================ CONFIG ================
JOB_ID = "batches/1oyi6zu7yfm8lvtuddedqtztq8vdfh56ox4t"
ORIGINAL_CSV_PATH = "../Data/API_Data/apps_test_data.csv"
FINAL_CSV_PATH = "finale.csv"


# ========================================

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Brak GEMINI_API_KEY w pliku .env")

    client = genai.Client(api_key=api_key)

    print(f"Sprawdzanie statusu zadania: {JOB_ID}...")

    # 1. Pobranie informacji o zadaniu z serwerów
    job = client.batches.get(name=JOB_ID)
    status_str = job.state.name if hasattr(job.state, 'name') else str(job.state)
    print(f"Obecny status: {status_str}")

    if "SUCCEEDED" not in status_str:
        print("Zadanie jeszcze się przetwarza. Uruchom skrypt ponownie za jakiś czas.")
        return

    print("Zadanie zakończone sukcesem! Pobieranie pliku z wynikami...")

    # 2. Pobranie pliku wynikowego
    try:
        result_file_name = job.dest.file_name
        file_content_bytes = client.files.download(file=result_file_name)
        file_content = file_content_bytes.decode('utf-8')
    except Exception as e:
        print(f"Błąd podczas pobierania pliku z serwerów: {e}")
        return

    print("Plik pobrany poprawnie. Przetwarzanie i wyciąganie danych...")
    results_list = []

    # 3. Parsowanie każdej linijki
    for line in file_content.splitlines():
        if not line.strip():
            continue

        try:
            data = json.loads(line)
            custom_id = data.get("id") or data.get("request", {}).get("id")
            if not custom_id:
                continue  # Pomijamy uszkodzone wiersze bez ID

            row_index = int(custom_id.replace("req-", ""))
            response_obj = data.get("response", {})
            candidates = response_obj.get("candidates", [])

            if candidates and "content" in candidates[0] and "parts" in candidates[0]["content"]:
                generated_text = candidates[0]["content"]["parts"][0].get("text", "")
                status = "success"
            else:
                generated_text = "Brak tekstu (prawdopodobnie blokada przez filtry bezpieczeństwa API)."
                status = "blocked_or_empty"

            results_list.append({
                "index": row_index,
                "status": status,
                "gemini_response": generated_text
            })

        except Exception as e:
            print(f"Błąd parsowania jednej z linii wyników: {e}")
            continue

    print(f"Pomyślnie wyciągnięto {len(results_list)} wyników. Łączenie z oryginalnym CSV...")

    # 4. Łączenie odzyskanych danych z oryginalnym plikiem Excel/CSV
    try:
        df_original = pd.read_csv(ORIGINAL_CSV_PATH)
        df_original = df_original.reset_index(drop=False)
        df_results = pd.DataFrame(results_list)

        if not df_results.empty:
            df_final = pd.merge(df_original, df_results, on="index", how="left")
            df_final.to_csv(FINAL_CSV_PATH, index=False, encoding="utf-8")
            print("=" * 60)
            print(f"🎉 GOTOWE! Ostateczne, połączone wyniki znajdziesz w pliku:")
            print(f"   ---> {FINAL_CSV_PATH}")
            print("=" * 60)
        else:
            print("Brak poprawnych wyników do zapisania.")

    except Exception as e:
        print(f"Błąd podczas operacji łączenia CSV: {e}")
        print("Zapis surowych, pobranych danych do pliku awaryjnego 'backup_results.jsonl'...")
        with open("backup_results.jsonl", "w", encoding="utf-8") as f:
            f.write(file_content)


if __name__ == "__main__":
    main()