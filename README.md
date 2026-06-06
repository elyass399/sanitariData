# 🩺 GDM Risk Predictor

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/license-MIT-green)

> Clinical decision support tool for **Gestational Diabetes Mellitus (GDM)** risk prediction — built on a real obstetric dataset of 10,000 patients.

---

## 🎯 Problem

Gestational Diabetes affects ~9% of pregnancies and is often underdiagnosed. Early detection from routine measurements can significantly improve maternal and fetal outcomes.

---

## ✨ Features

- 🧹 **Automated data cleaning** — clinical range validation, IQR clipping, logical consistency checks
- 📊 **EDA PDF report** — distributions, correlations, outlier analysis
- ⚙️ **GridSearchCV tuning** — finds best regularization via 5-fold cross-validation
- 🔌 **REST API** — Flask endpoint ready for integration
- 🖥️ **Minimal frontend** — plain HTML, no framework needed

---

## 📈 Results

| Metric | Value |
|---|---|
| Test ROC-AUC | **0.695** |
| Recall (diabetes class) | **0.63** |
| Cross-validation | 5-fold StratifiedKFold |
| Class imbalance handling | `class_weight="balanced"` |

**Top predictors (SHAP):** Serum Insulin → BMI → Age

---

## 🏗️ Architecture

```
dataset_en.csv
      ↓
eda.py        →  eda_report.pdf
train.py      →  models/model_gdm.joblib
      ↓
predict.py    →  reusable prediction module
app.py        →  REST API  :5000
index.html    →  frontend
```

---

## 📁 Structure

```
gdm-risk-predictor/
├── data/
│   └── dataset_en.csv
├── models/
│   └── model_gdm.joblib
├── eda.py
├── train.py
├── predict.py
├── app.py
├── index.html
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate EDA report
python eda.py

# Train model
python train.py

# Run API
python app.py
```

Open `index.html` in browser while `app.py` is running.

---

## 🔌 API

**POST** `/predict`

```json
// Request
{
  "Age": 35,
  "BMI": 32.0,
  "Serum_Insulin_2h": 180.0,
  "Diastolic_BP_mmHg": 80,
  "Previous_Pregnancies": 2,
  "Previous_C_Sections": 0
}

// Response
{
  "result": {
    "prediction": 1,
    "risk_score": 0.727,
    "label": "High Risk",
    "threshold": 0.5
  }
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML | scikit-learn · Logistic Regression |
| Tuning | GridSearchCV · StratifiedKFold |
| Explainability | SHAP LinearExplainer |
| API | Flask · flask-cors |
| Report | reportlab |
| Serialization | joblib |

---

## 👤 Author

**Elyass Rochdi** — AI & Data Science Engineer

[![GitHub](https://img.shields.io/badge/GitHub-elyass399-black?logo=github)](https://github.com/elyass399)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-elyass--rochdi-blue?logo=linkedin)](https://linkedin.com/in/elyass-rochdi)
