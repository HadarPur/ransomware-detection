import logging
import math
import os
import zlib

import numpy as np

from logger import setup_logging, get_logger

# Setup logging configuration to print to console
setup_logging(level=logging.INFO, log_to_file=False)
logger = get_logger(__name__)

def get_full_extension(file_name):
    # This splits the string at the first dot and takes everything after it
    parts = file_name.split('.')
    if len(parts) > 1:
        # returns everything from the first dot onwards: e.g., "DAT.csv.wk"
        return ".".join(parts[1:]).lower()
    return ""

def shannon_entropy(data):
    if not data:
        return 0
    probs = [data.count(b) / len(data) for b in set(data)]
    return -sum(p * math.log2(p) for p in probs)

def chunk_entropy(data, chunk_size=4096):
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    entropies = [shannon_entropy(chunk) for chunk in chunks if chunk]
    return np.mean(entropies), np.std(entropies)

def compression_ratio(data):
    return len(zlib.compress(data)) / max(len(data), 1)

def chi_square_uniform(data):
    if not data:
        return 0.0

    byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    expected = len(data) / 256

    chi_sq = np.sum((byte_counts - expected) ** 2 / expected)
    return chi_sq

def serial_byte_correlation(data):
    if len(data) < 2:
        return 0.0

    x = np.frombuffer(data, dtype=np.uint8).astype(np.float64)

    x1 = x[:-1]
    x2 = x[1:]

    mean = np.mean(x)

    numerator = np.sum((x1 - mean) * (x2 - mean))
    denominator = np.sum((x1 - mean) ** 2)

    if denominator == 0:
        return 0.0

    return numerator / denominator

def extract_features(filepath):
    with open(filepath, "rb") as f:
        data = f.read()

    file_name = os.path.basename(filepath)
    full_ext = get_full_extension(file_name)
    entropy = shannon_entropy(data)
    mean_ent, std_ent = chunk_entropy(data)
    com_rat = compression_ratio(data)
    zero_byte_ratio = data.count(b'\x00') / len(data)
    chi_sq = chi_square_uniform(data)
    chi_sq_norm = np.log1p(chi_sq)
    ser_byte_correlation = serial_byte_correlation(data)

    return {
        "file_name": file_name,
        "file_name_length": len(file_name),
        "file_size": len(data),
        "file_extension": full_ext,
        "entropy": entropy,
        "entropy_mean": mean_ent,
        "entropy_std": std_ent,
        "compression_ratio": com_rat,
        "zero_byte_ratio": zero_byte_ratio,
        "chi_square": chi_sq,
        "chi_square_normalized": chi_sq_norm,
        "serial_byte_correlation": ser_byte_correlation
    }


def pre_process_features(df):
    # Keep only rows where label matches the encryption status
    mask = ((df["valid_encryption"] == 1) & (df["label"] == "ENCRYPTED")) | \
           ((df["valid_encryption"] == 0) & (df["label"] == "CLEAN"))

    filtered_df = df.loc[mask].copy()

    # Identify dropped rows
    dropped_df = df.loc[~mask]
    if not dropped_df.empty:
        logger.info(f"Dropped {len(dropped_df)} files because they were invalid or mislabeled:")
        for _, row in dropped_df.iterrows():
            variant_info = f" \t | variant: {row['variant']}" if 'variant' in row else ""
            logger.info(f"  - {row['file_name']}{variant_info}")
    else:
        logger.info("No files were dropped.")

    # Drop column
    processed_df = filtered_df.drop(columns=["valid_encryption"])

    # Save
    processed_df.to_csv("encrypted_valid_only.csv", index=False)
    logger.info(f"Saved {len(processed_df)} rows to encrypted_valid_only.csv\n")

    return processed_df
