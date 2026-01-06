import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "entropy",
    "entropy_mean",
    "entropy_std",
    "compression_ratio",
    "zero_byte_ratio",
    "file_size",
    "file_name_length",
]

def rule_based_classifier(row):
    """
    Heuristic ransomware detector based on entropy and compression behavior.
    Returns:
        pred (int): 1 = encrypted, 0 = clean
        confidence (float): normalized score in [0,1]
    """
    score = 0

    if row["entropy"] > 7.2:
        score += 1
    if row["compression_ratio"] > 0.95:
        score += 1
    if row["entropy_std"] < 0.5:
        score += 1
    if row["zero_byte_ratio"] < 0.01:
        score += 1

    pred = 1 if score >= 3 else 0
    confidence = score / 4.0

    return pred, confidence

def fit_statistical_thresholds(df):
    """
    Fit statistical thresholds using CLEAN files only.
    """
    clean_df = df[df["label"] == "CLEAN"]

    thresholds = {
        "entropy": clean_df["entropy"].quantile(0.99),
        "compression_ratio": clean_df["compression_ratio"].quantile(0.99),
    }

    return thresholds

def statistical_classifier(row, entropy_thresh, compression_thresh):
    """
    Distribution-based detector using clean-file thresholds.
    """
    votes = 0

    if row["entropy"] > entropy_thresh:
        votes += 1
    if row["compression_ratio"] > compression_thresh:
        votes += 1

    pred = 1 if votes >= 1 else 0
    confidence = votes / 2.0

    return pred, confidence

def train_ml_classifier(df):
    """
    Train a lightweight ML classifier.
    Returns:
        model, scaler
    """
    X = df[FEATURES]
    y = (df["label"] == "ENCRYPTED").astype(int)

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    return model, scaler

def ml_predict(df, model, scaler):
    """
    Apply trained ML model to full DataFrame.
    Returns:
        probs (np.ndarray): encryption probabilities
        preds (np.ndarray): binary predictions
    """
    X_scaled = scaler.transform(df[FEATURES])
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return probs, preds

def ensemble_decision(row):
    """
    Combine all classifiers into final verdict.
    """
    score = (
        0.4 * row["rule_conf"] +
        0.3 * row["stat_conf"] +
        0.3 * row["ml_prob"]
    )

    pred = 1 if score >= 0.6 else 0
    return pred, score
