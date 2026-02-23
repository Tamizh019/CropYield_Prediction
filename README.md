---
title: AgriVision YieldMax
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
tags:
  - agriculture
  - machine-learning
  - crop-yield
  - xgboost
  - flask
  - ensemble
short_description: AI crop yield prediction with ensemble ML
env:
  GOOGLE_API_KEY:
    description: Google Gemini AI API key for agronomic insights
    required: false
---

# 🌾 AgriVision — YieldMax

**AI-Powered Crop Yield Prediction using Ensemble Learning**

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.12-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue)](https://www.docker.com/)

---

## What It Does

YieldMax predicts crop yields with **85–95% confidence** using a stacking ensemble:

| Model | Role |
|---|---|
| XGBoost (300 trees) | Categorical feature specialist |
| LightGBM (500 trees) | Environmental data expert |
| Deep Neural Network | Pattern detector |
| Ridge Meta-Learner | Intelligent weight optimizer |

**Key Features:**
- Single-page prediction form with smart auto-fill
- Confidence scores + prediction ranges (worst/expected/best)
- Bulk CSV upload for agricultural officers
- AI-powered agronomic insights via Gemini 2.0

---

## Quick Start (Local)

```bash
# 1. Clone & setup
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt

# 2. Create .env
GOOGLE_API_KEY=your_key
SECRET_KEY=your_secret

# 3. Train the ensemble (first-time only, ~25 min)
python scripts/train_ensemble.py

# 4. Run
python app.py
# Open http://localhost:5000
```

---

## Deploying to Hugging Face Spaces

1. Create a new **Docker** Space on Hugging Face
2. Push this repo to it:
   ```bash
   git remote add space https://huggingface.co/spaces/<username>/<space-name>
   git push space main
   ```
3. Set secrets in Space settings: `GOOGLE_API_KEY`, `SECRET_KEY`

> ⚠️ Upload model files (`models/*.pkl`) via the HF Files tab — they are excluded from git.

---

## Tech Stack

| Layer | Technologies |
|---|---|
| Backend | Flask 3.0, Python 3.10 |
| ML Models | XGBoost, LightGBM, TensorFlow 2.16 |
| ML Pipeline | Scikit-learn, Pandas, NumPy |
| AI Insights | Google Gemini 2.0 Flash |
| Frontend | Glassmorphism UI, Chart.js |
| Deployment | Docker, Gunicorn, HF Spaces |

---

## Project Structure

```
CropYield_Prediction/
├── app.py                   # Flask app
├── ensemble_model.py        # YieldMax ensemble class
├── Dockerfile               # HF Spaces deployment
├── requirements.txt
├── scripts/
│   └── train_ensemble.py    # Training pipeline
├── models/                  # Trained models (add manually)
├── templates/               # HTML pages
├── Datasets/
│   └── Crop_recommendation.csv
└── Docs/                    # Team documentation
```

---

## Team

- Arunmozhi Adithya
- Jenivaa
- Tamizharasan
- Pradeepraja
- Dilshan

---

**Resources:** [Gemini API Key](https://aistudio.google.com/app/apikey) · [Docs](./Docs/) · [Pre-trained Models](https://drive.google.com/drive/folders/1gMGjGMz0oCBkrMp2QtCyx14zPZ9lk4Y5)

*MIT License — Built for smarter, data-driven agriculture.*
