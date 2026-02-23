# 🏗️ AgriVision — Tech Stack & Architecture Guide

A plain-English walkthrough of how the entire system is built. Perfect for explaining during reviews or technical questions.

---

## 🧱 How the App Is Structured

Think of AgriVision as a three-floor building:

```
┌──────────────────────────────────┐
│  🖥️  Floor 3: Frontend (UI)      │  ← What the user sees
├──────────────────────────────────┤
│  ⚙️  Floor 2: Backend (Flask)    │  ← The brains that handle requests
├──────────────────────────────────┤
│  🤖  Floor 1: ML & AI Layer      │  ← The intelligence that predicts
└──────────────────────────────────┘
```

Each floor does its own job, but they all work together seamlessly.

---

## 🖥️ Floor 3 — The Frontend (What You See)

**HTML5 & CSS3** — We hand-built the UI using a modern **Glassmorphism** style (frosted glass cards, smooth gradients). No heavy frameworks — this keeps the app fast and lightweight.

**JavaScript** — Handles:
- Form validation (so users can't enter ridiculous values)
- Dynamic interactive elements (like the smart crop advisor wizard)
- Rendering prediction charts from Chart.js

> **Why no React/Vue?** For a data-driven Flask app like this, HTML templates are simpler, faster to load, and more than enough.

---

## ⚙️ Floor 2 — The Backend (Flask Server)

**Python 3.10+** — The entire backend and ML pipeline runs in Python.

**Flask** — A lightweight web framework that:
1. Serves the HTML pages to the user's browser
2. Receives form data when a farmer submits a prediction
3. Passes that data to the ML models
4. Returns the results back to the page

Think of Flask as the **waiter** — it takes your order, goes to the kitchen (ML layer), and brings back the food (prediction).

---

## 🤖 Floor 1 — The ML & AI Layer (The Intelligence)

This is where the real work happens. Three tools power the predictions:

### 🔢 Data Tools
| Tool | What It Does |
|---|---|
| **Pandas** | Loads and cleans the CSV training data |
| **NumPy** | Fast number crunching and math operations |
| **Scikit-learn** | Label encoding, Ridge Meta-Learner, Random Forest |

### 🌲 Yield Prediction Models (The Ensemble)
| Model | Speciality |
|---|---|
| **XGBoost** | Great with categorical data (crop types, states, seasons) |
| **LightGBM** | Fast, great with environmental numbers (temp, rainfall) |
| **TensorFlow / Keras DNN** | Finds hidden complex patterns across all features |
| **Ridge Meta-Learner** | Combines the three above intelligently |

### 🤝 AI Insights
**Google Gemini AI (Flash)** — After the ML models generate a number, we send that number to Gemini with a hidden prompt. Gemini acts as a virtual agronomist and returns plain-English farming advice tailored to the prediction.

> **Example:** Predicted 2,800 T/Ha with 80% humidity → Gemini responds: *"High humidity levels increase the risk of fungal diseases — consider preventive spraying."*

---

## 🔄 How It All Flows Together

Here is the complete journey from the moment a farmer clicks "Predict" to when they see their result:

```
1. Farmer fills form on the website
        ↓
2. Flask receives the form data
        ↓
3. Missing environmental data?
   → Yes: Auto-estimate from regional database
   → No: Use what the farmer entered
        ↓
4. Convert text to numbers (Label Encoding)
        ↓
5. Feed data into XGBoost + LightGBM + DNN (all at once)
        ↓
6. Ridge Meta-Learner combines the three predictions
        ↓
7. Calculate Confidence Score (how much models agreed)
        ↓
8. Send prediction to Google Gemini AI
        ↓
9. Gemini returns plain-English farming tips
        ↓
10. Flask renders everything on the results page 🎉
```

The entire process takes **under 2 seconds** for a live prediction.

---

## 🌱 The Two Main Features

### 1. Yield Predictor (YieldMax)
- Input: Location, Crop, Season, Area, Weather data
- Output: Tonnes/Hectare prediction + Confidence % + Worst/Expected/Best range
- Powered by: XGBoost + LightGBM + DNN → Ridge Meta-Learner

### 2. Smart Crop Advisor
- Input: Soil NPK values, Temperature, Humidity, Rainfall, pH
- Output: Top 3 recommended crops with match percentages
- Powered by: Random Forest Classifier (trained on 13 engineered features, balanced across crop types)

---

## 💬 One-Line Summary (For Reviews)

> *"We use a Python/Flask backend to run a stacking ensemble (XGBoost + LightGBM + Neural Network) for accurate yield predictions, and pipe the results into Google Gemini AI to turn raw numbers into practical farming advice."*

---

*AgriVision — Built smart, explained simply.* 🌾
