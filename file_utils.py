import hashlib
import logging
import os
import zipfile

import pandas as pd

from features_utils import extract_features
from logger import setup_logging, get_logger

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
    """
    Builds a features dataframe for CLEAN and ENCRYPTED datasets and assigns a
    'valid_encryption' flag.

    Semantics used here:
      - CLEAN rows: valid_encryption = 0
      - ENCRYPTED rows: valid_encryption = 1 by default (assumed encrypted because
        they come from Encrypted_Files), and set to 0 ONLY if we can prove the
        encrypted sample is byte-identical to its corresponding original.

    Key fixes vs. your version:
      1) ENCRYPTED rows default to 1, not 0, when we cannot pair by name (e.g., .thor GUID names).
      2) Match flags by a stable relative path (variant + filename) when possible,
         avoiding basename collisions across variants.
      3) Still supports your original name-substring pairing logic when filenames retain originals.
    """
    out_features = "features.csv"

    if os.path.exists(out_features):
        logger.skip(f"{out_features} already exists. Loading it...")
        df = pd.read_csv(out_features)
        logger.info(f"Found {len(df)} files.")
        return df

    # -----------------------
    # Read CLEAN
    # -----------------------
    logger.info("Reading CLEAN files...")
    original_df = read_file_data(clean_out, label="CLEAN")
    original_df["valid_encryption"] = 0  # CLEAN always 0
    logger.info(f"Found {len(original_df)} CLEAN files.")

    # -----------------------
    # Read ENCRYPTED
    # -----------------------
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

    # If there are no encrypted rows, just save/return the clean df.
    if encrypted_df.empty:
        df = original_df.copy()
        df.to_csv(out_features, index=False)
        logger.info(f"Saved {len(df)} rows to {out_features}")
        return df

    # -----------------------
    # Hash originals
    # -----------------------
    logger.info("Computing hash map of original files...")
    originals = collect_originals_with_hashes(clean_out)
    logger.info(f"{len(originals)} original files hashed.")

    # Build a fast lookup map: original filename -> original hash
    # originals items are assumed: orig_name -> (orig_path, orig_hash)
    orig_hash_by_name = {orig_name: orig_hash for orig_name, (_, orig_hash) in originals.items()}

    # -----------------------
    # Compute valid_encryption for encrypted files
    # -----------------------
    logger.info("Computing valid_encryption flags for encrypted files...")

    # We will map by relative path under encrypted_root to avoid basename collisions.
    # rel_key example: "10-Files/DAT.csv"
    valid_map = {}

    for dirpath, _, filenames in os.walk(encrypted_root):
        for enc_name in filenames:
            enc_path = os.path.join(dirpath, enc_name)
            rel_key = os.path.relpath(enc_path, encrypted_root).replace("\\", "/")

            # Default assumption: if it's in Encrypted_Files, it's encrypted.
            # We'll downgrade to 0 only if we can prove it's identical to a corresponding original.
            flag = 1

            # Try to pair encrypted file to an original via substring match
            # (your existing logic). This fails for GUID-renamed files like *.thor,
            # which is fine because default stays 1.
            matches = [orig_name for orig_name in orig_hash_by_name.keys() if orig_name in enc_name]

            if matches:
                try:
                    enc_hash = sha256_file(enc_path)
                except OSError:
                    logger.warn(f"Failed to read {enc_path}")
                    # Leave default flag=1; it's in encrypted set but unreadable
                    valid_map[rel_key] = flag
                    continue

                # If encrypted file is byte-identical to any matched original, it's not "valid encryption"
                if any(enc_hash == orig_hash_by_name[m] for m in matches):
                    flag = 0

            # Store final decision
            valid_map[rel_key] = flag

    # -----------------------
    # Assign flags to dataframe rows
    # -----------------------
    logger.info("Assigning valid_encryption flags to DataFrame rows...")
    encrypted_df = encrypted_df.copy()

    # IMPORTANT: This assumes read_file_data() gives you a file reference that can be
    # resolved to a path under encrypted_root. Because your original code uses basename,
    # I support both:
    #   - If 'file_name' is already a relative path under variant_dir, we can build rel_key.
    #   - Otherwise we fall back to variant + basename. If that still fails, default to 1.

    flags = []
    for _, row in encrypted_df.iterrows():
        fn = str(row.get("file_name", ""))

        # Determine variant folder if present in df
        variant = str(row.get("variant", "")).strip()

        # Build a stable rel_key candidate.
        # Common cases:
        #  - file_name is "DAT.csv" and variant is "10-Files"  -> "10-Files/DAT.csv"
        #  - file_name is already "10-Files/DAT.csv"          -> keep
        #  - file_name is a full path                          -> reduce to basename + variant
        fn_norm = fn.replace("\\", "/")

        if variant and ("/" not in fn_norm):
            rel_key = f"{variant}/{os.path.basename(fn_norm)}"
        else:
            # If it already looks like a relative path, try to use it.
            # If it's a full path, this will still include many segments; in that case fallback.
            rel_key = fn_norm

        # Best-effort: if rel_key not found and we have a variant, fallback to variant/basename
        if rel_key not in valid_map and variant:
            rel_key = f"{variant}/{os.path.basename(fn_norm)}"

        # Final default for encrypted files is 1 (assume encrypted)
        flags.append(valid_map.get(rel_key, 1))

    encrypted_df["valid_encryption"] = flags

    # -----------------------
    # Concatenate + save
    # -----------------------
    df = pd.concat([original_df, encrypted_df], ignore_index=True)
    df.to_csv(out_features, index=False)
    logger.info(f"Saved {len(df)} rows to {out_features}")

    return df

def extract_features_clean_only(clean_only_dir, out_features="features_clean_only.csv", dataset_tag="CLEAN_ONLY"):
    """
    Extract features ONLY for clean files, for false-positive evaluation.

    Output semantics:
      - label: "CLEAN"
      - valid_encryption: 0
      - dataset_tag: (optional string) to identify this dataset in later analysis

    This method does NOT touch encrypted_out and does NOT compute hashes/pairings.
    """

    if os.path.exists(out_features):
        logger.skip(f"{out_features} already exists. Loading it...")
        df = pd.read_csv(out_features)
        logger.info(f"Found {len(df)} clean-only files.")
        return df

    logger.info(f"Reading CLEAN-ONLY files from: {clean_only_dir}")
    df = read_file_data(clean_only_dir, label="CLEAN")  # assumes you already have this helper
    df = df.copy()

    logger.info(f"Found {len(df)} clean-only files.")
    df.to_csv(out_features, index=False)
    logger.info(f"Saved {len(df)} rows to {out_features}")

    return df

