import os
import zlib
import zipfile
import pandas as pd
from features_utils import extract_features
import hashlib
from logger import setup_logging, get_logger
import logging

# Setup logging configuration to print to console
setup_logging(level=logging.INFO, log_to_file=False)
logger = get_logger(__name__)

def extract_zip(zip_path: str, out_dir: str, password=None) -> str:
    """
    Extract a ZIP file to out_dir.
    If password is required, provide it as a string (e.g., "Password1").
    Returns the output directory.
    """
    os.makedirs(out_dir, exist_ok=True)

    logger.info("Start extracting: ", zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(
            path=out_dir,
            pwd=password.encode("utf-8") if password else None
        )

    for filename in os.listdir(out_dir):
        if filename.lower().endswith(".zip"):
            nested_zip = os.path.join(out_dir, filename)
            nested_out = os.path.join(out_dir, filename.replace(".zip", ""))

            os.makedirs(nested_out, exist_ok=True)

            with zipfile.ZipFile(nested_zip, "r") as nz:
                nz.extractall(
                    path=nested_out,
                    pwd=password.encode("utf-8") if password else None
                )

    logger.info("Extracted to:", out_dir)
    return out_dir

def extract_zip_if_needed(zip_path, out_dir, password=None):
    """
    Extracts a ZIP file only if the output directory does not already exist
    or is empty.
    """
    if os.path.exists(out_dir) and os.listdir(out_dir):
        logger.skip(f"{out_dir} already exists and is not empty.")
        return

    logger.info(f"Extracted: {zip_path} → {out_dir}")
    extract_zip(zip_path, out_dir, password=password)

def read_file_data(root_dir: str, label: str, variant=None) -> pd.DataFrame:
    """
    Walk through a directory and extract features from all files.
    Optionally attach ransomware variant metadata.
    """
    records = []

    for root, _, files in os.walk(root_dir):
        for name in files:
            path = os.path.join(root, name)
            try:
                feats = extract_features(path)
                feats["label"] = label
                feats["is_encrypted"] = 1 if label == 'ENCRYPTED' else 0
                if variant is not None:
                    feats["variant"] = variant
                records.append(feats)
            except Exception as e:
                logger.info(f"Skipping {path}: {e}")

    return pd.DataFrame(records)


def sha256_file(path, block_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

def collect_originals_with_hashes(original_root):
    """
    Return dict: orig_basename -> (full_path, sha256)
    Matches your working script exactly (keyed by basename).
    """
    originals = {}
    for dirpath, _, filenames in os.walk(original_root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                originals[name] = (path, sha256_file(path))
            except OSError:
                pass
    return originals

def compute_valid_encryption_for_encrypted_df(encrypted_df, encrypted_root, originals):
    """
    Adds valid_encryption to encrypted_df based on:
      - find originals whose basename is substring of encrypted file basename
      - if ANY such original hash == encrypted hash => valid_encryption=0
      - else valid_encryption=1
    If no matches => valid_encryption=0 (not validated / no relationship)
    """
    if encrypted_df is None or encrypted_df.empty:
        return encrypted_df

    if "file_name" not in encrypted_df.columns:
        raise KeyError(f"Expected 'file_name' in encrypted_df. Got: {list(encrypted_df.columns)}")

    orig_items = list(originals.items())  # [(orig_name, (orig_path, orig_hash)), ...]

    valid_flags = []
    for enc_rel in encrypted_df["file_name"].astype(str).tolist():
        enc_path = os.path.join(encrypted_root, enc_rel)

        # Fallback: if file_name is only basename but file is nested, try searching by basename
        if not os.path.exists(enc_path):
            base = os.path.basename(enc_rel)
            found = None
            for dirpath, _, filenames in os.walk(encrypted_root):
                if base in filenames:
                    found = os.path.join(dirpath, base)
                    break
            if found is None:
                valid_flags.append(0)
                continue
            enc_path = found

        enc_name = os.path.basename(enc_path)

        # matches: originals whose name appears in encrypted filename
        matches = [
            (orig_name, orig_hash)
            for orig_name, (_, orig_hash) in orig_items
            if orig_name in enc_name
        ]

        if not matches:
            valid_flags.append(0)
            continue

        try:
            enc_hash = sha256_file(enc_path)
        except OSError:
            valid_flags.append(0)
            continue

        identical = any(enc_hash == orig_hash for _, orig_hash in matches)
        valid_flags.append(0 if identical else 1)

    out = encrypted_df.copy()
    out["valid_encryption"] = valid_flags
    return out

def extract_features_from_files(clean_out, encrypted_out):
    out_features = "features.csv"

    if os.path.exists(out_features):
        logger.skip(f"{out_features} already exists. Loading it...\n")
        return pd.read_csv(out_features)

    logger.info("Reading CLEAN files...")
    original_df = read_file_data(clean_out, label="CLEAN")
    original_df["valid_encryption"] = 0  # CLEAN always 0
    logger.info(f"Found {len(original_df)} CLEAN files.")

    encrypted_root = os.path.join(encrypted_out, "Encrypted_Files")
    encrypted_parts = []

    logger.info(f"Scanning encrypted files under {encrypted_root}...")
    for d in sorted(os.listdir(encrypted_root)):
        variant_dir = os.path.join(encrypted_root, d)
        if os.path.isdir(variant_dir):
            part_df = read_file_data(variant_dir, label="ENCRYPTED", variant=d)
            logger.info(f"Variant '{d}': {len(part_df)} files")
            encrypted_parts.append(part_df)

    encrypted_df = pd.concat(encrypted_parts, ignore_index=True) if encrypted_parts else pd.DataFrame()
    logger.info(f"Total encrypted files: {len(encrypted_df)}")

    logger.info("Computing hash map of original files...")
    originals = collect_originals_with_hashes(clean_out)
    logger.info(f"{len(originals)} original files hashed.")

    # Compute valid_encryption for encrypted files
    valid_map = {}
    logger.info("Computing valid_encryption flags for encrypted files...")
    for dirpath, _, filenames in os.walk(encrypted_root):
        for enc_name in filenames:
            enc_path = os.path.join(dirpath, enc_name)
            matches = [
                (orig_name, orig_hash)
                for orig_name, (_, orig_hash) in originals.items()
                if orig_name in enc_name
            ]
            if not matches:
                continue

            try:
                enc_hash = sha256_file(enc_path)
            except OSError:
                logger.warn(f"Failed to read {enc_path}")
                continue

            identical = any(enc_hash == orig_hash for _, orig_hash in matches)
            valid_map[enc_path] = 0 if identical else 1

    logger.info(f"Assigning valid_encryption flags to DataFrame rows...")
    encrypted_df = encrypted_df.copy()
    flags = []
    for enc_rel in encrypted_df["file_name"].astype(str).tolist():
        base = os.path.basename(enc_rel)
        flag = next((v for p, v in valid_map.items() if os.path.basename(p) == base), 0)
        flags.append(flag)
    encrypted_df["valid_encryption"] = flags

    # Concatenate all data
    df = pd.concat([original_df, encrypted_df], ignore_index=True)
    df.to_csv(out_features, index=False)
    logger.info(f"Saved {len(df)} rows to {out_features}")

    return df
