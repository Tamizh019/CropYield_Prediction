# AgriVision v3.0 🌾🤖

> **⚠️ Welcome to the Deep Learning Branch!**  
> This branch uses **Deep Learning models** (CNN, LSTM) in addition to ML algorithms.  
> If you only need **Machine Learning models** (XGBoost, Random Forest), switch to the `main` branch.

**AI-Powered Agricultural Intelligence Platform**

Transform farming decisions with Machine Learning, Deep Learning, and real-time insights.

---

## 🚀 What's New in v3.0

| Feature | Technology | Description |
|---------|------------|-------------|
| 🩺 **Plant Doctor** | CNN (MobileNetV2) | Upload leaf photos → Instant disease diagnosis |
| 💰 **Market Prices** | LSTM Neural Network | 7-day crop price forecasting |
| 🌦️ **Weather Intelligence** | OpenWeatherMap API | Agricultural alerts & farming calendar |
| 🧪 **Fertilizer Calculator** | Optimization Algorithm | NPK-based cost-effective recommendations |

---

## 📋 Quick Start

> [!IMPORTANT]
> **New to the project?** 
> - If manual setup feels complicated, just double-click **`run.bat`**! It will handle the environment, dependencies, and start the app for you automatically.
> - For a deep understanding of the system, models, and API, check out our **[Docs/ Directory](./Docs/README.md)**.

### 1. Setup API Keys
Create a `.env` file:
```env
GOOGLE_API_KEY=your_gemini_api_key
OPENWEATHER_API_KEY=your_openweather_key  # Optional
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train Models (First Time)
```bash
python train_models.py          # ML models (Yield, Recommendation)
```

### 3.1 Setup Plant Disease Dataset (Deep Learning)
Required for `train_disease_model.py`.
1. Download **PlantVillage** dataset from Kaggle:
   - **Full Dataset (Recommended):** [PlantVillage Dataset (2.18 GB)](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) - Better accuracy.
   - **Lightweight (Fast):** [PlantDisease (342 MB)](https://www.kaggle.com/datasets/emmarex/plantdisease) - Quicker training.
2. Extract it into `Datasets/PlantVillage`.
3. Run the organization script:
```bash
python organize_dataset.py      # Fixes folder structure & class names
python train_disease_model.py   # Trains the CNN model
```

### 3.2 Train Market Price Model (Optional)
Enable the real LSTM forecasting model:
```bash
python train_price_model.py
```

### 4. Run the App
```bash
python app.py
```
Open `http://127.0.0.1:5000`

---

## 🧠 Technology Stack

### Machine Learning
- **XGBoost & Random Forest** - Yield prediction & crop recommendation
- **MobileNetV2 CNN** - Plant disease detection (Transfer Learning)
- **LSTM RNN** - Time-series price forecasting

### Backend
- **Flask** - Web framework
- **TensorFlow/Keras** - Deep Learning
- **Gemini 2.0 Flash** - AI insights generation

### Frontend
- **Glassmorphism UI** - Modern design
- **Chart.js** - Data visualization
- **Responsive** - Mobile-friendly

---

## 📁 Project Structure

```
CropYield_Prediction/
├── app.py                      # Main Flask application
├── train_models.py             # ML model training
├── train_disease_model.py      # CNN training script
├── organize_dataset.py         # Dataset helper script
├── disease_detection.py        # Plant Doctor module
├── price_forecast.py           # LSTM price prediction
├── weather_service.py          # Weather API integration
├── fertilizer_optimizer.py     # NPK calculator
├── models/                     # Trained models (.pkl, .h5)
├── templates/                  # HTML templates
│   ├── index.html
│   ├── plant_doctor.html
│   ├── market_prices.html
│   ├── weather.html
│   └── fertilizer.html
├── Datasets/                   # Training data
└── static/                     # CSS, JS, images
```

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

- Disease detection works in "mock mode" without trained CNN
- Price forecasting uses simulation when LSTM model is not trained

---

**Built with ❤️**
