import os
import pandas as pd
from file_utils import extract_zip_if_needed, extract_features_from_files
from visualization_utils import feature_visualization

CLEAN_FILES_PATH = "./files/Original_Files.zip"
ENCRYPTED_FILES_PATH = "./files/Encrypted_Files_2.zip"
VALIDATION_FILES_PATH = "./files/More_Clean_Files.zip" # for validation

def main():
    base_out = "extracted_files"
    clean_out = os.path.join(base_out, "clean")
    encrypted_out = os.path.join(base_out, "encrypted")
    validation_out = os.path.join(base_out, "validation")

    print("----------- Zip extraction -----------")
    extract_zip_if_needed(CLEAN_FILES_PATH, clean_out)
    extract_zip_if_needed(ENCRYPTED_FILES_PATH, encrypted_out, password="Password1")
    extract_zip_if_needed(VALIDATION_FILES_PATH, validation_out)

    print("\n----------- Features extraction -----------")
    df = extract_features_from_files(clean_out, encrypted_out)

    print("\n----------- Create Plots For Visualization -----------")
    feature_visualization(df)

if __name__ == "__main__":
    main()
