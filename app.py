import os
import pandas as pd
from file_utils import extract_zip_if_needed, extract_features_from_files
from data_exploration import run_exploratory_visualizations
from features_utils import pre_process_features
from visualization_utils import plot_model_evaluation
from detector import RansomwareDetector
from evaluation import evaluate_model_performance
from logger import setup_logging, get_logger
import logging

files_to_extract = False
CLEAN_FILES_PATH = "./files/Original_Files.zip"
ENCRYPTED_FILES_PATH = "./files/Encrypted_Files_2.zip"
VALIDATION_FILES_PATH = "./files/More_Clean_Files.zip" # for validation

# Setup logging configuration to print to console
setup_logging(level=logging.INFO, log_to_file=False)

logger = get_logger(__name__)
logger.info("Application started - Ransomware Detection\n")

def main():
    base_out = "extracted_files"
    clean_out = os.path.join(base_out, "clean")
    encrypted_out = os.path.join(base_out, "encrypted")
    validation_out = os.path.join(base_out, "validation")

    logger.info("----------- Zip Extraction -----------")
    if files_to_extract:
        extract_zip_if_needed(CLEAN_FILES_PATH, clean_out)
        extract_zip_if_needed(ENCRYPTED_FILES_PATH, encrypted_out, password="Password1")
        extract_zip_if_needed(VALIDATION_FILES_PATH, validation_out)
    else:
        logger.skip("skipping zip extraction\n")

    logger.info("----------- Features Extraction -----------")
    df = extract_features_from_files(clean_out, encrypted_out)

    logger.info("----------- Pre Processing -----------")
    df = pre_process_features(df)

    logger.info("----------- Feature Rationale (Exploration) -----------")
    run_exploratory_visualizations(df)

    logger.info("----------- Run Models -----------")
    # Initialize the class we built
    detector = RansomwareDetector()

    # Train the three classification approaches (RF, LR, KNN)
    detector.train(df)

    logger.info("----------- Performance Evaluation -----------")
    # Evaluate on the training set or a split
    y_pred_labels = detector.evaluate_batch(df)
    # Convert labels back to 0/1 for the matrix
    y_pred_numeric = y_pred_labels.map({'ENCRYPTED': 1, 'NOT ENCRYPTED': 0})
    # Generate the performance plots
    plot_model_evaluation(df['is_encrypted'], y_pred_numeric)
    # evaluate_model_performance(detector, df)

if __name__ == "__main__":
    main()
