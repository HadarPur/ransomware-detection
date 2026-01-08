import os
import pandas as pd
from file_utils import extract_zip_if_needed, extract_features_from_files
from visualization_utils import feature_visualization
from detector import RansomwareDetector
from evaluation import evaluate_model_performance

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
    print(df.head(40))

    print("\n----------- Create Plots For Visualization -----------")
    feature_visualization(df)

    print("\n----------- Run Models -----------")
    # Initialize the class we built
    detector = RansomwareDetector()

    # Train the three classification approaches (RF, LR, KNN)
    detector.train(df)

    print("\n----------- Performance Evaluation -----------")
    # Evaluate on the training set or a split
    evaluate_model_performance(detector, df)

if __name__ == "__main__":
    main()
