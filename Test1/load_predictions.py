import pandas as pd
import pickle


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_lines(assert_data):
    """
    Converts PKL entry to clean list of lines.
    Handles:
    - list[str]
    - multiline string
    - None
    """

    if assert_data is None:
        return []

    if isinstance(assert_data, str):
        return assert_data.splitlines()

    if isinstance(assert_data, list):
        lines = []
        for item in assert_data:
            if item is None:
                continue
            lines.extend(str(item).splitlines())
        return lines

    return []


def process_assertions(assert_data, fn_name):

    lines = normalize_lines(assert_data)

    valid = []
    seen = set()
    duplicates = 0

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if line.startswith("assert"):
            candidate = line

        elif line.startswith(fn_name):
            candidate = "assert " + line

        else:
            continue

        if candidate in seen:
            duplicates += 1
            continue

        seen.add(candidate)
        valid.append(candidate)

    return "\n".join(valid), duplicates


def merge_files(csv_path, pkl_path, output_path):

    df = pd.read_csv(csv_path)

    pkl_data = load_pkl(pkl_path)

    asserts_col = []
    duplicates_col = []

    for i, row in df.iterrows():

        fn_name = str(row["fn_name"]).strip()

        assert_data = pkl_data[i] if i < len(pkl_data) else []

        asserts, duplicates = process_assertions(assert_data, fn_name)

        asserts_col.append(asserts)
        duplicates_col.append(duplicates)

    df["PyTester_asserts"] = asserts_col
    df["PyTester_duplicates"] = duplicates_col

    df.to_csv(output_path, index=False)

    print("Saved:", output_path)


if __name__ == "__main__":

    CSV_FILE = "PyTester_More_tests.csv"
    PKL_FILE = "predictions.pkl"

    OUTPUT_FILE = "PyTester_with_asserts.csv"

    merge_files(CSV_FILE, PKL_FILE, OUTPUT_FILE)