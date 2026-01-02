import os
import pandas as pd
from file_utils import extract_zip_if_needed, read_file_data

ORIGINAL_FILES_PATH = "./files/Original_Files.zip"
ENCRYPTED_FILES_PATH = "./files/Encrypted_Files_2.zip"
CLEAN_FILES_PATH = "./files/More_Clean_Files.zip"

# ---------- Main pipeline ----------

def main():
    base_out = "extracted_files"
    original_out = os.path.join(base_out, "original")
    encrypted_out = os.path.join(base_out, "encrypted")
    clean_out = os.path.join(base_out, "clean")

    # Encrypted zip is password-protected per assignment ("Password1")
    print("----------- Zip extraction -----------")
    extract_zip_if_needed(ORIGINAL_FILES_PATH, original_out)
    extract_zip_if_needed(ENCRYPTED_FILES_PATH, encrypted_out, password="Password1")
    extract_zip_if_needed(CLEAN_FILES_PATH, clean_out)

    print("\n----------- Features extraction -----------")
    # Build DataFrames
    original_df = read_file_data(original_out, label="ORIGINAL")

    # For encrypted, also capture variant from subdirectory name (e.g., "1-Files", "2-Files", ...)
    encrypted_parts = []
    for d in sorted(os.listdir(encrypted_out)):
        variant_dir = os.path.join(encrypted_out, d)
        if os.path.isdir(variant_dir):
            encrypted_parts.append(read_file_data(variant_dir, label="ENCRYPTED", variant=d))

    encrypted_df = pd.concat(encrypted_parts, ignore_index=True) if encrypted_parts else pd.DataFrame()

    df = pd.concat([original_df, encrypted_df], ignore_index=True)

    print(df.head(10))
    df.to_csv("features.csv", index=False)
    print(f"\nSaved features to features.csv with {len(df)} rows.")


if __name__ == "__main__":
    main()
