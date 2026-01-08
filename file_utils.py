import os
import zlib
import zipfile
import pandas as pd
from features_utils import extract_features

def extract_zip(zip_path: str, out_dir: str, password=None) -> str:
    """
    Extract a ZIP file to out_dir.
    If password is required, provide it as a string (e.g., "Password1").
    Returns the output directory.
    """
    os.makedirs(out_dir, exist_ok=True)

    print("Start extracting: ", zip_path)

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

    print("Extracted to:", out_dir)
    return out_dir

def extract_zip_if_needed(zip_path, out_dir, password=None):
    """
    Extracts a ZIP file only if the output directory does not already exist
    or is empty.
    """
    if os.path.exists(out_dir) and os.listdir(out_dir):
        print(f"[SKIP] {out_dir} already exists and is not empty.")
        return

    print(f"[EXTRACTING] {zip_path} → {out_dir}")
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
                print(f"Skipping {path}: {e}")

    return pd.DataFrame(records)

def extract_features_from_files(clean_out, encrypted_out):
    out_features = "features.csv"

    if os.path.exists(out_features):
        print(f"[SKIP] {out_features} already exists and is not empty.")
        df = pd.read_csv(out_features)
        return df


    # Build DataFrames
    original_df = read_file_data(clean_out, label="CLEAN")

    # For encrypted, also capture variant from subdirectory name (e.g., "1-Files", "2-Files", ...)
    encrypted_root = os.path.join(encrypted_out, "Encrypted_Files")
    encrypted_parts = []

    for d in sorted(os.listdir(encrypted_root)):
        variant_dir = os.path.join(encrypted_root, d)

        if not os.path.isdir(variant_dir):
            continue

        encrypted_parts.append(
            read_file_data(
                variant_dir,
                label="ENCRYPTED",
                variant=d
            )
        )

    encrypted_df = (
        pd.concat(encrypted_parts, ignore_index=True)
        if encrypted_parts else pd.DataFrame()
    )

    df = pd.concat([original_df, encrypted_df], ignore_index=True)

    df.to_csv("features.csv", index=False)
    print(f"\nSaved features to features.csv with {len(df)} rows.")

    return df