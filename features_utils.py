import os
import math
import zlib
import numpy as np
import pandas as pd
from visualization_utils import visualization

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

def extract_features(filepath):
    with open(filepath, "rb") as f:
        data = f.read()

    entropy = shannon_entropy(data)
    mean_ent, std_ent = chunk_entropy(data)
    file_name = os.path.basename(filepath)

    _, file_extension = os.path.splitext(file_name)

    chi_sq = chi_square_uniform(data)
    chi_sq_normalized = chi_sq / len(data)

    return {
        "file_name": file_name,
        "file_name_length": len(file_name),
        "file_size": len(data),
        "file_extension": file_extension.lower(),
        "entropy": entropy,
        "entropy_mean": mean_ent,
        "entropy_std": std_ent,
        "compression_ratio": compression_ratio(data),
        "zero_byte_ratio": data.count(b'\x00') / len(data),
        "chi_square": chi_sq,
        "chi_sq_normalized": chi_sq_normalized,
    }