import logging
import os
import pandas as pd

from sklearn.model_selection import train_test_split

from data_exploration import run_exploratory_visualizations
from detector import RansomwareDetector
from features_utils import pre_process_features
from file_utils import extract_zip_if_needed, extract_features_from_files, extract_features_clean_only, split_clean_and_save
from logger import setup_logging, get_logger
from model_evaluation import plot_model_evaluation
from assesst_false_positive import run_false_positive_assessment

files_to_extract = True
CLEAN_FILES_PATH = "./files/Original_Files.zip"
ENCRYPTED_FILES_PATH = "./files/Encrypted_Files_2.zip"
VALIDATION_FILES_PATH = "./files/More_Clean_Files.zip"  # for validation

# Setup logging configuration to print to console
setup_logging(level=logging.INFO, log_to_file=False)

logger = get_logger(__name__)
logger.info("Application started - Ransomware Detection\n")

def split_data(df, feature_cols, label_col='is_encrypted',
               test_size=0.2, random_state=42):
    """
    Splits dataframe into stratified train/test sets.
    """
    X = df[feature_cols]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


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
    mixed_df = extract_features_from_files(clean_out, encrypted_out)
    clean_only_df = extract_features_clean_only(validation_out)

    logger.warning("mixed_df source rows: %d", len(mixed_df))
    logger.warning("clean_only_df source rows: %d", len(clean_only_df))

    logger.info("----------- Balance Dataset -----------")
    clean_val_df, mixed_augmented_df = split_clean_and_save(
        clean_only_df=clean_only_df,
        mixed_df=mixed_df,
        output_dir="data/balanced"
    )

    logger.info("len(mixed_augmented_df): %d", len(mixed_augmented_df))
    logger.info("len(clean_val_df): %d", len(clean_val_df))

    logger.info("----------- Pre Processing -----------")
    training_df = pre_process_features(mixed_augmented_df)

    logger.warning(
        "TRAIN DATA SIZE = %d | encrypted=%d clean=%d",
        len(training_df),
        training_df['is_encrypted'].sum(),
        len(training_df) - training_df['is_encrypted'].sum()
    )

    logger.info("----------- Feature Rationale (Exploration) -----------")
    run_exploratory_visualizations(training_df)

    logger.info("----------- Run Models -----------")
    # Initialize the class we built
    detector = RansomwareDetector()

    # Split the data
    X_train, X_test, y_train, y_test = split_data(
        training_df,
        feature_cols=detector.feature_cols,
        label_col='is_encrypted',
        test_size=0.2
    )

    # Train the three classification approaches (ad, LR, KNN)
    detector.train(X_train, y_train)

    logger.info("----------- Models Evaluation -----------")

    train_ensemble, train_ad, train_lr, train_knn = detector.evaluate(
        X_train, y_train, dataset_name="Train"
    )

    train_dir = os.path.join("model_plots", "train")
    plot_model_evaluation(y_train, train_ad, out_dir=train_dir,
                          title="Train - ad Confusion Matrix", file_prefix="AdaBoostClassifier")
    plot_model_evaluation(y_train, train_lr, out_dir=train_dir,
                          title="Train - Logistic Regression Confusion Matrix", file_prefix="LogisticRegression")
    plot_model_evaluation(y_train, train_knn, out_dir=train_dir,
                          title="Train - KNN Confusion Matrix", file_prefix="KNeighborsClassifier")

    plot_model_evaluation(y_train, train_ensemble, out_dir=train_dir,
                          title="Train - Ensemble Confusion Matrix", file_prefix="Ensemble")

    test_ensemble, test_ad, test_lr, test_knn = detector.evaluate(
        X_test, y_test, dataset_name="Test"
    )

    test_dir = os.path.join("model_plots", "test")
    plot_model_evaluation(y_test, test_ad, out_dir=test_dir,
                          title="Test - ad Confusion Matrix", file_prefix="AdaBoostClassifier")
    plot_model_evaluation(y_test, test_lr, out_dir=test_dir,
                          title="Test - Logistic Regression Confusion Matrix", file_prefix="LogisticRegression")
    plot_model_evaluation(y_test, test_knn, out_dir=test_dir,
                          title="Test - KNeighborsClassifier Confusion Matrix", file_prefix="KNeighborsClassifier")

    plot_model_evaluation(y_test, test_ensemble, out_dir=test_dir,
                          title="Test - Ensemble Confusion Matrix", file_prefix="Ensemble")

    logger.info("----------- Assess False Positive -----------")
    run_false_positive_assessment(detector, clean_val_df)

if __name__ == "__main__":
    main()
