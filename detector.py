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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

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

    def train(self, X_train, y_train):
        total_files = len(y_train)
        encrypted_count = int(y_train.sum())
        logger.info(f"Train files: {total_files} | Encrypted: {encrypted_count} | Clean: {total_files - encrypted_count}")

        # Feature stats on training only
        stats = X_train.groupby(y_train).mean()
        logger.info(f"Training feature means per class:\n{stats}\n")

        X_train_scaled = self.scaler.fit_transform(X_train)

        self.svc_model.fit(X_train_scaled, y_train)
        self.lr_model.fit(X_train_scaled, y_train)
        self.knn_model.fit(X_train_scaled, y_train)

        logger.info("Models trained successfully.\n")

    def predict_batch_labels(self, X):
        X_scaled = self.scaler.transform(X)

        svc_pred = self.svc_model.predict(X_scaled).astype(int)
        lr_pred  = self.lr_model.predict(X_scaled).astype(int)
        knn_pred = self.knn_model.predict(X_scaled).astype(int)

        votes_sum = svc_pred + lr_pred + knn_pred
        # majority vote: 1 if at least 2 models predict 1
        ensemble_pred = (votes_sum >= 2).astype(int)

        return ensemble_pred, svc_pred, lr_pred, knn_pred

    def evaluate(self, X_test, y_test, dataset_name="Test"):
        ensemble_pred, svc_pred, lr_pred, knn_pred = self.predict_batch_labels(X_test)

        logger.info(f"------- Overall {dataset_name} Dataset Results -------")
        total_files = len(y_test)
        encrypted_count = int(y_test.sum())
        logger.info(f"Test files: {total_files} | Encrypted: {encrypted_count} | Clean: {total_files - encrypted_count}")

        logger.info(f"Accuracy: {accuracy_score(y_test, ensemble_pred):.4f}")
        logger.info(f"Precision: {precision_score(y_test, ensemble_pred):.4f}")
        logger.info(f"Recall: {recall_score(y_test, ensemble_pred):.4f}")
        logger.info(f"F1-score: {f1_score(y_test, ensemble_pred):.4f}")

        logger.info(f"Confusion matrix:\n{confusion_matrix(y_test, ensemble_pred)}")
        logger.info(f"Classification report:\n{classification_report(y_test, ensemble_pred, digits=4)}")

        # Optional: per-model metrics (often useful)
        logger.info(f"------- {dataset_name} results (SVC) ------- ")
        logger.info(f"\n{classification_report(y_test, svc_pred, digits=4)}")

        logger.info(f"------- {dataset_name} results (LogisticRegression) -------")
        logger.info(f"\n{classification_report(y_test, lr_pred, digits=4)}")

        logger.info(f"------- {dataset_name} results (KNeighborsClassifier) -------")
        logger.info(f"\n{classification_report(y_test, knn_pred, digits=4)}")

        return ensemble_pred, svc_pred, lr_pred, knn_pred
