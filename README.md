# 🌾 AgriVision v3.1

**AI-Powered Agricultural Intelligence Platform**

> Transform farming decisions with Machine Learning, Deep Learning, and real-time AI insights.

---

## ⚡ Quick Start

**Option 1: One-Click Run**
```
Double-click run.bat
```

**Option 2: Manual Setup**
```bash
# 1. Create .env file with API keys
GOOGLE_API_KEY=your_gemini_api_key
OPENWEATHER_API_KEY=your_weather_key  # Optional

# 2. Install & Run
pip install -r requirements.txt
python train_models.py   # First time only
python app.py            # Start server
```

Open **http://127.0.0.1:5000**

---

## 🎯 Features

| Module | Technology | What it Does |
|--------|------------|--------------|
| 📊 **Yield Prediction** | XGBoost | Predict crop yields with bulk CSV upload |
| 🌱 **Crop Recommendation** | Random Forest | Find best crops for your land |
| 🩺 **Plant Doctor** | CNN (MobileNetV2) | Diagnose plant diseases from photos |
| 💰 **Market Prices** | LSTM | 7-day price forecasting |
| 🌦️ **Weather Intel** | OpenWeatherMap API | Farming alerts & 5-day forecast |
| 🧪 **Fertilizer Calc** | Optimization | NPK-based recommendations |
| 🤖 **AI Insights** | Gemini 2.0 Flash | Smart analysis on every page |

---

## 🆕 What's New (v3.1)

### ML Analytics Dashboard
- **Model Confidence Score** - Shows prediction reliability
- **Feature Importance Chart** - Which factors affect yield most
- **Yield Distribution Histogram** - Visualize prediction ranges
- **Prediction Classification** - High/Medium/Low yield breakdown

### AI Farming Advisor
- **Actionable Recommendations** - Priority actions, not just data description
- **Risk Mitigation** - Potential issues & solutions
- **Growth Opportunities** - Where to expand cultivation

### Bug Fixes
- Fixed state name display (was showing codes, now shows names)
- Improved page spacing for better readability
- Reduced table preview to 10 rows for cleaner UI

---

## 🧠 Tech Stack

| Layer | Technologies |
|-------|--------------|
| **ML Models** | XGBoost, Random Forest, MobileNetV2 CNN, LSTM |
| **Backend** | Flask, TensorFlow/Keras, Gemini 2.0 |
| **Frontend** | Glassmorphism UI, Chart.js |
| **APIs** | OpenWeatherMap, Google Gemini |

---

## 📁 Project Structure

```
CropYield_Prediction/
├── app.py                 # Main Flask app (all routes)
├── train_models.py        # ML model training
├── train_disease_model.py # CNN training
├── disease_detection.py   # Plant Doctor module
├── price_forecast.py      # LSTM predictions
├── weather_service.py     # Weather API
├── models/                # Trained .pkl & .h5 files
├── templates/             # HTML pages
├── static/                # CSS, JS, images
├── Datasets/              # Training data
└── Docs/                  # Full documentation
```

---

## 📚 Documentation

See **[Docs/README.md](./Docs/README.md)** for:
- System architecture & data flow
- ML/DL model explanations
- API reference
- Training guides

---

## 👥 Team

- Arunmozhi Adithya
- Jenivaa
- Tamizharasan
- Pradeepraja
- Dilshan

---

## 📝 Notes

> [!TIP]
> **Don't want to train?** If your PC is low-spec or you want to skip training, use our pre-trained models:
> 
> 📥 **[Download Pre-Trained Models (Google Drive)](https://drive.google.com/drive/folders/1gMGjGMz0oCBkrMp2QtCyx14zPZ9lk4Y5)**
> 
> Extract files into the `models/` folder and you're ready to go!

---

**Built with ❤️**
