# 🛡️ AI-Powered Network Intrusion Detection System (NIDS) Using Suricata

This project demonstrates a machine learning-based approach to detecting network intrusions using Suricata alerts. It extracts meaningful features from Suricata logs, trains both Random Forest and XGBoost classifiers, and provides tools for inference and model evaluation.

---

## 📁 Project Structure

```text
ai-nids-suricata/
├── models/
│   ├── random_forest.pkl
│   └── xgb_classifier.pkl
├── train_suricata_from_file.py
├── infer_suricata_event.py
├── evaluate_suricata_models.py
├── feature_columns.json
├── label_encoder.pkl
└── alerts-only.json
```




---

## 🚀 Features

- Parses large Suricata alert datasets in JSON format
- Extracts flow, HTTP, and core fields
- Encodes features and trains:
  - ✅ Random Forest (Scikit-learn)
  - ✅ XGBoost Classifier
- Allows single-event inference for predictions
- Evaluation script provides F1-score and classification metrics

---

## 📥 Large Files (Hosted on Google Drive)

Due to GitHub file size limits, large data and model files are hosted externally:

| File | Description | Google Drive Link |
|------|-------------|------------------|
| `alerts-only.json` | Full Suricata alert dataset | [Download](<https://https://drive.google.com/file/d/1uh5Owj6iX03uKl2MyRXjK0LngzcEDcK4/view?usp=sharing>) |
| `models/random_forest.pkl` | Trained Random Forest model | [Download](<https://drive.google.com/file/d/1uh5Owj6iX03uKl2MyRXjK0LngzcEDcK4/view?usp=drive_link>) |

*Above files too large to be hosted on github

---

## 📦 Requirements

- Python 3.10+
- scikit-learn
- xgboost
- pandas
- numpy
- joblib

To install dependencies:

```bash
pip install -r requirements.txt

## Training
python train_suricata_from_file.py
#This will predict the alert signature using both trained models on a hardcoded example or passed input.

## Evaluation
python infer_suricata_event.py

Runs the trained models against the full dataset and outputs performance metrics.


🎯 Example Output
🟩 Random Forest Prediction: ET SCAN Nmap User-Agent Detected
🟦 XGBoost Prediction: SURICATA HTTP URI invalid

📄 License
This project is licensed under the MIT License.

