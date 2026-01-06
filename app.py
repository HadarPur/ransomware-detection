import os
import pandas as pd
from file_utils import extract_zip_if_needed, extract_features_from_files
from visualization_utils import feature_visualization
from classifiers import rule_based_classifier, statistical_classifier, train_ml_classifier, ml_predict, ensemble_decision

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

    print("\n----------- Prepare Labels and Fit Models  -----------")
    df["is_encrypted"] = (df["label"] == "ENCRYPTED").astype(int)

    # Statistical thresholds
    clean_df = df[df["label"] == "CLEAN"]
    entropy_thresh = clean_df["entropy"].quantile(0.99)
    compression_thresh = clean_df["compression_ratio"].quantile(0.99)

    # ML model
    ml_model, scaler = train_ml_classifier(df)

    print("\n----------- Apply classifiers -----------")
    df[["rule_pred", "rule_conf"]] = df.apply(
        lambda r: pd.Series(rule_based_classifier(r)), axis=1
    )

    df[["stat_pred", "stat_conf"]] = df.apply(
        lambda r: pd.Series(
            statistical_classifier(r, entropy_thresh, compression_thresh)
        ),
        axis=1
    )

    df["ml_prob"], df["ml_pred"] = ml_predict(df, ml_model, scaler)
    df[["final_pred", "final_conf"]] = df.apply(
        lambda r: pd.Series(ensemble_decision(r)), axis=1
    )

    print(df.head(40))

if __name__ == "__main__":
    main()
