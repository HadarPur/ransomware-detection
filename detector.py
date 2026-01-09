import numpy as np
import pandas as pd
import logging
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from logger import setup_logging, get_logger
import logging
from sklearn.metrics import confusion_matrix, classification_report

# Setup logging configuration to print to console
setup_logging(level=logging.INFO, log_to_file=False)
logger = get_logger(__name__)

class RansomwareDetector:
    def __init__(self):
        # Approach A: SVC
        self.svc_model = SVC(kernel='rbf', C=0.1, gamma='scale', probability=True)
        # Approach B: Logistic Regression
        self.lr_model = LogisticRegression(C=0.01, class_weight='balanced')
        # Approach C: K-Nearest Neighbors
        self.knn_model = KNeighborsClassifier(n_neighbors=5)

        self.scaler = StandardScaler()
        self.feature_cols = ['entropy', 'entropy_mean', 'entropy_std', 'compression_ratio', 'zero_byte_ratio', 'chi_square_normalized', 'serial_byte_correlation']

    def train(self, df):
        """
        Trains all three models and logs data statistics.
        """
        # Log dataset composition
        total_files = len(df)
        encrypted_count = df['is_encrypted'].sum()
        logger.info(
            f"Total files: {total_files} | Encrypted: {encrypted_count} | Clean: {total_files - encrypted_count}")

        X = df[self.feature_cols]
        y = df['is_encrypted']

        # Log feature averages to understand the "Static Features"
        stats = X.groupby(y).mean()
        logger.info(f"Feature means per class:\n{stats}\n")

        X_scaled = self.scaler.fit_transform(X)

        self.svc_model.fit(X_scaled, y)
        self.lr_model.fit(X_scaled, y)
        self.knn_model.fit(X_scaled, y)

        logger.info("Models trained successfully using SVC, Logistic Regression and K-Nearest Neighbors.")

    def predict_file(self, file_features_dict, verbose=False):
        """
        Predicts a single file and optionally logs model agreement.
        """
        features_df = pd.DataFrame([[file_features_dict[col] for col in self.feature_cols]], columns=self.feature_cols)
        features_scaled = self.scaler.transform(features_df)

        # Get individual model votes
        votes = [
            int(self.svc_model.predict(features_scaled)[0]),
            int(self.lr_model.predict(features_scaled)[0]),
            int(self.knn_model.predict(features_scaled)[0])
        ]

        encrypted_count = sum(votes)
        verdict = "ENCRYPTED" if encrypted_count >= 2 else "NOT ENCRYPTED"
        confidence = (encrypted_count / 3) if verdict == "ENCRYPTED" else (3 - encrypted_count) / 3

        if verbose:
            logger.debug(
                f"File: {file_features_dict.get('file_name', 'Unknown')} | Votes: {votes} | Verdict: {verdict}")

        return {
            "verdict": verdict,
            "confidence_score": confidence,
            "votes": votes
        }

    def evaluate_batch(self, df, dataset_name="Testing"):
        """
        Runs batch prediction, logs overall performance metrics, saves results.
        """
        # Run predictions for each row
        results = df.apply(lambda row: self.predict_file(row.to_dict()), axis=1)
        verdicts = [r['verdict'] for r in results]

        # Calculate full model agreement (3/3 votes)
        confidences = [r['confidence_score'] for r in results]
        full_agreement = sum(1 for c in confidences if c == 1.0)

        logger.info(f"Full Model Consensus (3/3 votes): {full_agreement} / {len(df)} files")

        # Save predictions with model votes to CSV
        predictions_df = pd.DataFrame(results.tolist())
        final_output = pd.concat([df.reset_index(drop=True), predictions_df], axis=1)
        output_path = "ransomware_detection_results.csv"
        final_output.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}\n")

        # Return verdicts and metrics if needed
        return pd.Series(verdicts)
