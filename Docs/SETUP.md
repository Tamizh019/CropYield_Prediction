# 🚀 AgriVision Setup Guide

Complete installation and configuration guide for developers.

---

## Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.9 - 3.11 | Core runtime |
| pip | Latest | Package manager |
| Git | Any | Version control |
| 8GB+ RAM | - | For training models |

---

## Step 1: Clone & Navigate

```bash
git clone <your-repo-url>
cd CropYield_Prediction
```

---

## Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Key Packages
| Package | Purpose |
|---------|---------|
| `flask` | Web framework |
| `tensorflow` | Deep learning (CNN, LSTM) |
| `xgboost` | Yield prediction model |
| `scikit-learn` | ML utilities |
| `pandas`, `numpy` | Data processing |
| `google-generativeai` | Gemini AI integration |

---

## Step 4: Environment Variables

Create a `.env` file in the root directory:

```env
# Required
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional (for live weather)
OPENWEATHER_API_KEY=your_openweather_key_here

# Flask
SECRET_KEY=your_random_secret_key
```

### Getting API Keys
1. **Gemini AI**: [Google AI Studio](https://aistudio.google.com/app/apikey)
2. **OpenWeather**: [OpenWeatherMap API](https://openweathermap.org/api)

---

## Step 5: Dataset Setup (For Plant Doctor)

1. Download the PlantVillage dataset from [Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
2. Extract to `Datasets/PlantVillage/`
3. Run the organization script:

```bash
python organize_dataset.py
```

This creates the required `train/` and `val/` folder structure.

---

## Step 6: Train Models (Optional)

If models are not pre-trained:

```bash
# Train Yield + Recommendation models
python train_models.py

# Train Plant Disease CNN
python train_disease_model.py

# Train Price LSTM
python train_price_model.py
```

> [!TIP]
> **Low-Spec PC?** Training deep learning models (CNN, LSTM) can take 1-2 hours and requires 8GB+ RAM.
> 
> **Alternative:** Download our pre-trained models instead!
> 
> 📥 **[Download Pre-Trained Models (Google Drive)](https://drive.google.com/drive/folders/1gMGjGMz0oCBkrMp2QtCyx14zPZ9lk4Y5)**
> 
> Simply extract the files into the `models/` folder and skip training!

---

## Step 7: Run the Application

```bash
python app.py
```

Open your browser: **http://127.0.0.1:5000**

---

## Folder Structure

```
CropYield_Prediction/
├── app.py                 # Main Flask application
├── models/                # Trained model files (.pkl, .h5)
├── templates/             # HTML templates
├── static/                # CSS, JS, images
├── Datasets/              # Training data (gitignored)
├── Docs/                  # Documentation
├── .env                   # Environment variables (gitignored)
└── requirements.txt       # Python dependencies
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `GOOGLE_API_KEY not found` | Create `.env` file with your key |
| `Dataset not found` | Run `python organize_dataset.py` |
| TensorFlow GPU issues | Install `tensorflow-cpu` instead |

---

## Next Steps

- Read [ARCHITECTURE.md](./ARCHITECTURE.md) to understand the system
- Check [API_REFERENCE.md](./API_REFERENCE.md) for API endpoints
- Review [MODELS_OVERVIEW.md](./MODELS_OVERVIEW.md) for ML details
