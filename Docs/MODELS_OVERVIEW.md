# 🤖 Models Overview

Detailed explanation of all Machine Learning and Deep Learning models in AgriVision.

---

## Summary Table

| Model | Type | Algorithm | Input | Output | File |
|-------|------|-----------|-------|--------|------|
| Yield Prediction | ML | XGBoost | Crop, Location, Climate | tonnes/hectare | `yield_model.pkl` |
| Crop Recommendation | ML | Random Forest | NPK, Climate | Best Crop | `recommend_model.pkl` |
| Disease Detection | DL | CNN (MobileNetV2) | Leaf Image | Disease Name | `disease_model.h5` |
| Price Forecast | DL | LSTM | Historical Prices | 7-Day Prices | `price_lstm.h5` |

---

## 1. Yield Prediction Model (XGBoost)

### What It Does
Predicts crop yield (tonnes/hectare) based on environmental and location factors.

### Input Features
| Feature | Type | Example |
|---------|------|---------|
| State_Name | Categorical | "Tamil Nadu" |
| District_Name | Categorical | "Coimbatore" |
| Crop | Categorical | "Rice" |
| Area | Numeric | 5.5 hectares |
| Temperature | Numeric | 28°C |
| Humidity | Numeric | 75% |
| Rainfall | Numeric | 150mm |
| pH | Numeric | 6.5 |

### Why XGBoost?
- Handles mixed data types (categorical + numeric)
- Fast training and prediction
- Built-in feature importance
- Robust to outliers

### Training
```bash
python train_models.py
```

---

## 2. Crop Recommendation Model (Random Forest)

### What It Does
Recommends the most suitable crop based on soil nutrients and climate.

### Input Features
| Feature | Type | Range |
|---------|------|-------|
| N (Nitrogen) | Numeric | 0-140 ppm |
| P (Phosphorus) | Numeric | 0-145 ppm |
| K (Potassium) | Numeric | 0-205 ppm |
| Temperature | Numeric | 10-45°C |
| Humidity | Numeric | 20-100% |
| pH | Numeric | 3.5-10 |
| Rainfall | Numeric | 20-300mm |

### Output
One of 22 crops: Rice, Wheat, Maize, Cotton, Sugarcane, etc.

### Why Random Forest?
- Excellent for multi-class classification
- Handles feature interactions well
- No overfitting with proper tuning

---

## 3. Disease Detection Model (CNN)

### What It Does
Identifies plant diseases from leaf images using computer vision.

### Architecture
```
Input Image (224x224x3)
        ↓
┌───────────────────┐
│   MobileNetV2     │  ← Pre-trained on ImageNet (Transfer Learning)
│   (Base Model)    │
│   Frozen Weights  │
└─────────┬─────────┘
          ↓
┌───────────────────┐
│  Global Average   │
│     Pooling       │
└─────────┬─────────┘
          ↓
┌───────────────────┐
│   Dense (256)     │  ← New trainable layer
│   ReLU + Dropout  │
└─────────┬─────────┘
          ↓
┌───────────────────┐
│   Dense (N)       │  ← N = number of disease classes
│   Softmax         │
└───────────────────┘
```

### Supported Classes
- Tomato (10 classes: Healthy, Early Blight, Late Blight, etc.)
- Potato (3 classes)
- Corn (4 classes)

### Why Transfer Learning?
- MobileNetV2 already "knows" edges, textures, shapes
- We only train the final layers for our specific diseases
- Achieves 90%+ accuracy with just 10,000 images

### Training
```bash
python train_disease_model.py
```

---

## 4. Price Forecast Model (LSTM)

### What It Does
Predicts future commodity prices using time-series analysis.

### Architecture
```
Input: 60-day price history
        ↓
┌───────────────────┐
│   LSTM (128)      │  ← Learns temporal patterns
└─────────┬─────────┘
          ↓
┌───────────────────┐
│   LSTM (64)       │  ← Deeper pattern extraction
└─────────┬─────────┘
          ↓
┌───────────────────┐
│   Dense (25)      │
│   Dense (1)       │  ← Single price output
└───────────────────┘
```

### Why LSTM?
- Designed for sequential/time-series data
- "Remembers" long-term price trends
- Handles seasonality and market cycles

### Training Data
Uses real market data from:
- `price_dataset_One.csv` (daily mandi prices)
- `price_dataset_Two.csv` (monthly averages)

### Training
```bash
python train_price_model.py
```

---

## Model Performance

| Model | Metric | Score |
|-------|--------|-------|
| Yield (XGBoost) | R² Score | ~0.85 |
| Recommend (RF) | Accuracy | ~95% |
| Disease (CNN) | Accuracy | ~92% |
| Price (LSTM) | MAPE | ~8% |

---

## Retraining Guidelines

1. **Yield/Recommend**: Retrain when new crop data is available
2. **Disease**: Add new disease classes by expanding the dataset
3. **Price**: Retrain monthly with fresh market data
