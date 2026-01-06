# Ransomware Detection (Feature Extraction + Classifiers)

Description
- Small Python project to extract features from files (clean / encrypted), visualize features, train classifiers (rule-based, statistical thresholds, ML) and produce ensemble decisions.

Prerequisite
- Edit the file paths at the top of `app.py` to point to your zip files before running:
  - `./files/Original_Files.zip`
  - `./files/Encrypted_Files_2.zip`
  - `./files/More_Clean_Files.zip`

🧱 Environment Setup
- Create and activate a virtual environment:
```
    python -m venv rd-env
    source rd-env/bin/activate     \# Linux/Mac
    rd-env\Scripts\activate        \# Windows
```
- Install required packages:
```
    pip install biopython pandas scikit-learn matplotlib openai
```

🏗 System Architecture
- `main.py` — orchestrates extraction, feature extraction, visualization, training, and prediction.
- `file_utils.py` — zip extraction and feature extraction helpers.
- `visualization_utils.py` — plotting functions for features.
- `classifiers.py` — rule-based classifier, statistical thresholding, ML training and prediction, ensemble logic.
- Output directories:
  - `extracted_files/clean`
  - `extracted_files/encrypted`
  - `extracted_files/validation`

🔄 High-Level Workflow
1. Unzip provided archives into `extracted_files/*` folders.
2. Extract features from each file (entropy, compression ratio, size, etc.).
3. Visualize feature distributions and relationships.
4. Label data (`CLEAN` vs `ENCRYPTED`) and compute statistical thresholds from clean files.
5. Train an ML classifier (features scaled) and compute prediction probabilities.
6. Apply three classifiers:
   - Rule-based
   - Statistical thresholds (entropy / compression)
   - Machine learning model
7. Combine results via an ensemble decision and output final predictions.

▶️ Running the Application
1. Ensure file paths in `app.py` are set to your zip files.
2. Activate the virtual environment (see Environment Setup).
3. Run the app:
```
    python app.py
```
4. Expected results:
   - `extracted_files/` populated with unzipped content.
   - Printed pandas DataFrame head showing features and predictions.
   - Plots saved or displayed (as implemented in `visualization_utils.py`).
   - Trained ML model and scaler available in memory (or persisted if implemented).
