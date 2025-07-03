# 🛡️ AI-Powered Network Intrusion Detection System (NIDS) Using Suricata

This project demonstrates a machine learning-based approach to detecting network intrusions using Suricata alerts. It extracts meaningful features from Suricata logs, trains both Random Forest and XGBoost classifiers, and provides tools for inference and model evaluation.

---

## 📁 Project Structure

```
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

| File | Description | Access |
|------|-------------|--------|
| `alerts-only.json` | Full Suricata alert dataset | File available upon request |
| `models/random_forest.pkl` | Trained Random Forest model | File available upon request |

*These files are too large to be hosted directly on GitHub.*

---

## 🧰 Installation Guide

### 1. 📦 Install Python

#### Linux/macOS:
```bash
sudo apt update && sudo apt install python3 python3-pip -y
```

#### Windows:
Download Python 3.10+ from: https://www.python.org/downloads/

Then verify:
```bash
python --version
pip --version
```

---

### 2. 📦 Install Dependencies

Clone this repository and navigate into it:

```bash
git clone https://github.com/your_username/ai-nids-suricata.git
cd ai-nids-suricata
```

Install Python packages:

```bash
pip install -r requirements.txt
```

---

## 🧪 Usage

### 📊 Training

```bash
python train_suricata_from_file.py
```

Trains both Random Forest and XGBoost classifiers using the provided alert dataset.

---

### 🔎 Inference (Single Event)

```bash
python infer_suricata_event.py --input path_to_alert.json
```

Classifies a single Suricata event. If no input is provided, a built-in test example is used.

---

### 🧾 Evaluation (Full Dataset)

```bash
python evaluate_suricata_models.py
```

Runs both models against the full dataset and outputs F1-score and classification metrics.

---

## 🎯 Example Output

```
🟩 Random Forest Prediction: ET SCAN Nmap User-Agent Detected
🟦 XGBoost Prediction: SURICATA HTTP URI invalid
```

---

## 📄 License

This project is licensed under the MIT License.
