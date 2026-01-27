# Crop Yield Prediction System 🌾
Hey Guys! This is our machine learning project that helps farmers decide what to grow and how much they can expect to harvest.

## Work in Progress 🚧
We are still working on this project! We will add more features soon.

## Team Members 👥
- **Arunmozhi Adithya**
- **Jenivaa**
- **Pradeepraja**
- **Dilshan**

## What's New? (v2.0 Updates) 🚀
We have upgraded the system with the latest tech:
- **Agri-Intelligence AI:** Now powered by **Gemini 2.0 Flash** to give smart advice and risk analysis.
- **Advanced Models:** We now use **Ensemble Learning** (Random Forest + XGBoost + Gradient Boosting) to get the best possible accuracy.
- **Bulk Analytics:** Upload improved CSVs and get detailed reports instantly.

## How to run it?
### 🔑 IMPORTANT: API Key Setup
Before running, you **must** setup the AI:
1. Create a file named `.env` in this folder.
2. Add your Google Gemini API key inside it like this:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
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
