# Ransomware Detection System

A machine learning-based ransomware detection system that analyzes file characteristics to identify encrypted files. The system extracts statistical features from files and uses an ensemble of machine learning classifiers to detect potential ransomware encryption.

## Overview

This project implements a comprehensive ransomware detection pipeline that:

- Extracts multiple statistical features from files (entropy, compression ratio, byte patterns, etc.)
- Trains and evaluates multiple machine learning classifiers
- Uses ensemble voting for robust predictions
- Provides comprehensive evaluation metrics and visualizations
- Assesses false positive rates on clean files

The system is designed to identify encrypted files by analyzing patterns that distinguish encrypted content from normal files, which is useful for detecting ransomware activity before files are encrypted.

## Features

### Feature Extraction
The system extracts the following features from each file:
- **Entropy**: Shannon entropy measuring randomness in file content
- **Entropy Statistics**: Mean and standard deviation of entropy across file chunks
- **Compression Ratio**: Ratio of compressed size to original size (encrypted files compress poorly)
- **Zero Byte Ratio**: Proportion of null bytes in the file
- **Chi-Square Test**: Statistical measure of byte distribution uniformity
- **Serial Byte Correlation**: Correlation between consecutive bytes
- **File Metadata**: Size, extension, and filename characteristics

### Machine Learning Models
The system employs three classifiers with ensemble voting:
1. **AdaBoost Classifier**: Uses decision stumps as weak learners
2. **Logistic Regression**: Linear classifier with balanced class weights
3. **K-Nearest Neighbors**: Instance-based classifier

Final predictions use majority voting from all three models.

## Project Structure

```
ransomware-detection/
├── app.py                      # Main application orchestration
├── detector.py                 # RansomwareDetector class with ML models
├── features_utils.py           # Feature extraction and preprocessing
├── file_utils.py               # File I/O, ZIP extraction, data processing
├── data_exploration.py         # Exploratory data analysis and visualizations
├── model_evaluation.py         # Model evaluation metrics and plots
├── assesst_false_positive.py   # False positive assessment
├── logger.py                   # Logging utilities
├── files/                      # Input ZIP files
│   ├── Original_Files.zip
│   ├── Encrypted_Files_2.zip
│   └── More_Clean_Files.zip
├── extracted_files/            # Extracted files
│   ├── clean/                  # Clean (unencrypted) files
│   ├── encrypted/              # Encrypted files (multiple variants)
│   └── validation/             # Validation dataset
├── data/                       # Processed datasets
│   ├── pre-processed/          # Raw extracted features
│   ├── balanced/               # Balanced training datasets
│   └── processed/              # Final processed features
├── data_exploration_plots/     # EDA visualizations
└── model_plots/                # Model evaluation plots
    ├── train/                  # Training set results
    ├── test/                   # Test set results
    └── validation/             # Validation set results
```

## Requirements

- **Python**: 3.9 or higher
- **Operating System**: macOS (development tested), Linux/Windows compatible
- **Dependencies**:
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - numpy

## Installation

### 1. Clone or Download the Repository

```bash
cd ransomware-detection
```

### 2. Create and Activate Virtual Environment

**macOS/Linux:**
```bash
python -m venv rd-env
source rd-env/bin/activate
```

**Windows:**
```bash
python -m venv rd-env
rd-env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install pandas scikit-learn matplotlib seaborn numpy
```

Or install from requirements file (if available):
```bash
pip install -r requirements.txt
```

## Configuration

Before running the application, configure the file paths in `app.py`:

```python
files_to_extract = False  # Set to True to extract ZIP files
CLEAN_FILES_PATH = "./files/Original_Files.zip"
ENCRYPTED_FILES_PATH = "./files/Encrypted_Files_2.zip"
VALIDATION_FILES_PATH = "./files/More_Clean_Files.zip"
```

**Note**: If you want to extract files from ZIP archives:
- Set `files_to_extract = True`
- Ensure the ZIP files are in the `files/` directory
- The encrypted files ZIP is password-protected (default: "Password1")

## Usage

### Running the Full Pipeline

```bash
python app.py
```

This will execute the complete pipeline:

1. **File Extraction** (if `files_to_extract = True`)
   - Extracts clean files from `Original_Files.zip`
   - Extracts encrypted files from `Encrypted_Files_2.zip`
   - Extracts validation files from `More_Clean_Files.zip`

2. **Feature Extraction**
   - Processes all files and extracts statistical features
   - Validates encryption by comparing file hashes
   - Creates balanced training datasets

3. **Data Preprocessing**
   - Filters valid encrypted files
   - Balances the dataset
   - Prepares features for training

4. **Exploratory Data Analysis**
   - Generates visualizations of feature distributions
   - Creates correlation matrices and pair plots
   - Saves plots to `data_exploration_plots/`

5. **Model Training**
   - Trains AdaBoost, Logistic Regression, and KNN classifiers
   - Splits data into train/test sets (80/20)
   - Uses stratified splitting to maintain class balance

6. **Model Evaluation**
   - Evaluates models on training and test sets
   - Generates confusion matrices and classification reports
   - Creates evaluation plots in `model_plots/train/` and `model_plots/test/`

7. **False Positive Assessment**
   - Tests models on clean validation files
   - Assesses false positive rates
   - Generates validation plots

### Output

The application generates:

- **Processed Data**: CSV files in `data/` directories
- **Visualizations**: 
  - Feature exploration plots in `data_exploration_plots/`
  - Model evaluation plots in `model_plots/`
- **Console Output**: Detailed logs with metrics, confusion matrices, and classification reports

## Workflow Details

### Feature Extraction Process

For each file, the system:
1. Reads file content as binary data
2. Computes Shannon entropy and chunk-based entropy statistics
3. Calculates compression ratio using zlib
4. Measures zero byte ratio
5. Performs chi-square test for byte distribution uniformity
6. Computes serial byte correlation between consecutive bytes
7. Extracts file metadata (name, size, extension)

### Data Validation

The system validates encrypted files by:
- Computing SHA-256 hashes for all files
- Matching encrypted files to their original clean counterparts
- Flagging files as "valid_encryption" only if they differ from originals
- Filtering out corrupted or mislabeled files

### Ensemble Decision Making

The final prediction uses majority voting:
- Each model (AdaBoost, Logistic Regression, KNN) votes on classification
- If 2 or more models predict "encrypted", the file is classified as encrypted
- This approach improves robustness and reduces false positives

## Model Performance

The system evaluates models using standard classification metrics:
- **Accuracy**: Overall classification correctness
- **Precision**: Ratio of true positives to all predicted positives
- **Recall**: Ratio of true positives to all actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions vs. actual labels

Results are displayed in the console and saved as visualizations.

## Troubleshooting

### Common Issues

1. **ZIP Extraction Fails**
   - Ensure ZIP files are in the correct location
   - Check file permissions
   - Verify encrypted ZIP password is correct

2. **Memory Issues**
   - Process files in smaller batches
   - Ensure sufficient RAM for large datasets

3. **Missing Dependencies**
   - Activate virtual environment before running
   - Install all required packages

4. **File Path Errors**
   - Verify file paths in `app.py` are correct
   - Use absolute paths if relative paths fail

## Development

### Key Components

- **RansomwareDetector** (`detector.py`): Main detector class with ML models
- **Feature Extraction** (`features_utils.py`): Statistical feature computation
- **File Processing** (`file_utils.py`): ZIP handling and data management
- **Visualization** (`data_exploration.py`, `model_evaluation.py`): Plotting utilities

### Extending the System

To add new features:
1. Implement feature computation in `features_utils.py`
2. Add feature name to `detector.feature_cols` in `detector.py`
3. Retrain models to incorporate new features

To add new models:
1. Import and initialize model in `RansomwareDetector.__init__()`
2. Train model in `RansomwareDetector.train()`
3. Add prediction to `RansomwareDetector.predict_batch_labels()`
4. Update ensemble voting logic if needed

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

This system demonstrates practical application of machine learning for cybersecurity, specifically in detecting ransomware encryption patterns through statistical file analysis.
