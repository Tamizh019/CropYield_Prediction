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
short_description: AI-powered crop yield prediction using XGBoost, LightGBM & DNN ensemble
env:
  GOOGLE_API_KEY:
    description: Google Gemini AI API key for agronomic insights
    required: false
---

# 🌾 AgriVision — YieldMax Precision Model

**AI-Powered Crop Yield Prediction with Ensemble Learning**

Advanced machine learning system combining XGBoost, LightGBM, and Deep Neural Networks for accurate, confident crop yield predictions.

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.12-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green)](https://flask.palletsprojects.com/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue)](https://www.docker.com/)

---

## 🚀 Deploy on Hugging Face Spaces

This app is ready to deploy as a **Docker Space** on [Hugging Face Spaces](https://huggingface.co/spaces).

### Steps to Deploy

1. **Create a new Space** on Hugging Face → choose **Docker** as the SDK.
2. **Push this repository** to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/agrivision-yieldmax
   git push space main
   ```
3. **Set Secret Environment Variables** in your Space settings:
   - `GOOGLE_API_KEY` → your [Google Gemini API key](https://aistudio.google.com/app/apikey) *(optional – needed for AI insights)*
   - `SECRET_KEY` → any random string for Flask session security

> **Note:** Pre-trained model files (`yieldmax_ensemble.pkl`, etc.) must be present in the `models/` folder before deployment. Upload them via Git LFS or the HF dataset hub.

---

## Quick Start (Local)

### Option 1: One-Click Run (Windows)
```bash
# Double-click this file
run.bat
```

### Option 2: Manual Setup
```bash
# 1. Create virtual environment
python -m venv venv
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # Linux/Mac

# 2. Create .env file with API keys
echo GOOGLE_API_KEY=your_key > .env
echo SECRET_KEY=your_secret >> .env

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train YieldMax ensemble (first time only, ~25 min)
python scripts/train_ensemble.py

# 5. Start server
python app.py
```

**Open:** [http://localhost:5000](http://localhost:5000)

### Option 3: Docker (Local)
```bash
# Build image
docker build -t agrivision-yieldmax .

# Run container
docker run -p 7860:7860 \
  -e GOOGLE_API_KEY=your_key \
  -e SECRET_KEY=your_secret \
  agrivision-yieldmax

# Open http://localhost:7860
```

---

## What is YieldMax?

**YieldMax Precision Model** is an ensemble learning system that predicts crop yields with **85-95% confidence** by combining:

- **XGBoost** (300 trees) — Categorical feature specialist
- **LightGBM** (500 trees) — Environmental data expert
- **Deep Neural Network** (256→128→64→32→1) — Pattern detector
- **Ridge Meta-Learner** — Intelligent weight optimizer (Stacking)

### Key Features

**Smart Input System**
- Single-page form (no complex wizard)
- Auto-estimates environmental conditions if you don't have soil data
- Real-time defaults shown as placeholders

**Reliable Predictions**
- Unified ensemble output (not confusing multiple models)
- Confidence scores (0-100%) based on model agreement
- Prediction ranges (worst/expected/best scenarios)

**Dual-Mode Display**
- **Production Mode:** Clean, farmer-friendly output
- **Technical Mode:** Detailed breakdown for presentations (`?technical=true`)

**Enterprise-Ready**
- Bulk CSV upload for agricultural officers
- Analytics dashboard with charts
- AI-powered agronomic insights (Gemini 2.0)

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Backend** | Flask 3.0, Python 3.10 |
| **Ensemble Models** | XGBoost 2.0, LightGBM 4.1, TensorFlow 2.16 |
| **ML Pipeline** | Scikit-learn, Pandas, NumPy |
| **AI Insights** | Google Gemini 2.0 Flash |
| **Frontend** | Glassmorphism UI, JavaScript, Chart.js |
| **Deployment** | Docker, Gunicorn, Hugging Face Spaces |

**Optimized Dependencies:** Only 12 core packages
```
Flask, pandas, numpy, scikit-learn
xgboost, lightgbm, tensorflow
google-generativeai, markdown
python-dotenv, Werkzeug, gunicorn
```

---

## Project Structure

```
CropYield_Prediction/
├── app.py                         # Main Flask application
├── ensemble_model.py              # YieldMax Ensemble class
├── requirements.txt               # Core dependencies
├── Dockerfile                     # Docker image for HF Spaces
├── .env                           # API keys (create manually)
├── run.bat                        # One-click startup (Windows)
│
├── scripts/
│   ├── train_ensemble.py          # YieldMax training pipeline
│   └── train_models.py            # Legacy fallback models
│
├── models/                        # Trained models (after training)
│   ├── yieldmax_ensemble.pkl      # Main ensemble (~150MB)
│   ├── yield_label_encoders.pkl   # Categorical encoders
│   ├── yield_features.pkl         # Feature metadata
│   └── ensemble_metadata.pkl      # Training metadata
│
├── templates/                     # HTML pages
│   ├── base.html                  # Base template
│   ├── index.html                 # Landing page
│   ├── predict_yield.html         # YieldMax predictor
│   └── recommend.html             # Crop recommendation
│
├── Datasets/                      # Training data
│   └── Yield_Data_With_Environment.csv  # Required dataset
│
├── Docs/                          # Team Documentation
│   ├── PRESENTATION_GUIDE.md      # Simple guide for team review
│   └── ARCHITECTURE_AND_TECH_STACK.md  # How the ML pipeline & tech stack works
│
└── logs/                          # Training logs (generated)
```

---

## Team

- Arunmozhi Adithya
- Jenivaa
- Tamizharasan
- Pradeepraja
- Dilshan

---

## License

MIT License - Feel free to use for educational purposes.

---

## Resources

**Get Gemini API Key:** [Google AI Studio](https://aistudio.google.com/app/apikey)  
**Documentation:** [Docs folder](./Docs/)  
**Pre-trained Models (optional):** [Google Drive](https://drive.google.com/drive/folders/1gMGjGMz0oCBkrMp2QtCyx14zPZ9lk4Y5)

---

## Project Highlights

- **Ensemble Learning** (Advanced technique, production-grade)
- **245k+ Training Records** (Real agricultural data)
- **92% Accuracy** (R² score, outperforms single models)
- **AI-Powered Insights** (Gemini 2.0 Flash integration)
- **User-Friendly** (Smart auto-fill, clean interface)
- **Transparent** (Technical mode for auditing)
- **Well-Documented** (4 comprehensive guides)
- **Docker Ready** (One-command deployment to HF Spaces)

---

**Built for smarter, data-driven agriculture**

*Making farming decisions scientific, one prediction at a time.*
