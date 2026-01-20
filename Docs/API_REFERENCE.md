# 🔌 API Reference

Complete documentation of all Flask routes and endpoints.

---

## Base URL

```
http://127.0.0.1:5000
```

---

## 🌐 Web Pages

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Dashboard homepage |
| `/predict_yield` | GET/POST | Yield prediction form |
| `/recommend_crop` | GET/POST | Crop recommendation form |
| `/plant_doctor` | GET/POST | Disease detection upload |
| `/market_prices` | GET/POST | Price forecasting |
| `/weather` | GET/POST | Weather information |
| `/fertilizer` | GET/POST | Fertilizer calculator |
| `/documentation` | GET | Help documentation |
| `/sample_data` | GET | Download CSV templates |

---

## 🔗 API Endpoints

### 1. Model Status

```http
GET /api/model_status
```

**Response:**
```json
{
  "yield_model": true,
  "recommend_model": true,
  "ai_model": true,
  "prediction_count": 42,
  "price_model": {
    "status": "Active",
    "type": "LSTM Neural Net"
  },
  "weather_service": {
    "status": "Active",
    "type": "Live API"
  },
  "disease_model": true
}
```

---

### 2. Price Forecast API

```http
POST /api/forecast
Content-Type: application/json
```

**Request Body:**
```json
{
  "crop": "Rice",
  "state": "Tamil Nadu"
}
```

**Response:**
```json
{
  "success": true,
  "crop": "Rice",
  "current_price": 2450.50,
  "forecast": [
    {"date": "2026-01-21", "price": 2460.00},
    {"date": "2026-01-22", "price": 2475.00}
  ],
  "trend": "increasing",
  "data_source": "LSTM Model"
}
```

---

### 3. Disease Detection API

```http
POST /api/detect_disease
Content-Type: multipart/form-data
```

**Request:**
- `image`: File (JPEG/PNG of plant leaf)

**Response:**
```json
{
  "success": true,
  "disease": "Tomato_Early_Blight",
  "confidence": 0.94,
  "treatment": "Apply copper-based fungicide...",
  "prevention": "Rotate crops, remove infected leaves..."
}
```

---

### 4. Weather API

```http
POST /api/weather
Content-Type: application/json
```

**Request Body:**
```json
{
  "city": "Chennai"
}
```

**Response:**
```json
{
  "success": true,
  "city": "Chennai",
  "current": {
    "temp": 32,
    "humidity": 78,
    "description": "Partly Cloudy"
  },
  "forecast": [...]
}
```

---

## 🛡️ Error Handling

All API endpoints return errors in this format:

```json
{
  "success": false,
  "error": "Error message here"
}
```

| HTTP Code | Meaning |
|-----------|---------|
| 200 | Success |
| 400 | Bad Request (missing/invalid params) |
| 500 | Server Error |

---

## 📊 Bulk Prediction (CSV Upload)

```http
POST /predict_yield
Content-Type: multipart/form-data
```

**Request:**
- `file`: CSV file with columns:
  - State_Name, District_Name, Crop, Area
  - Temperature, Humidity, Rainfall, pH

**Response:** Redirects to `/bulk_result` with:
- Predictions for each row
- Summary statistics
- AI-generated insights
- Downloadable results CSV
