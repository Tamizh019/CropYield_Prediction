# 🧠 How Deep Learning Works in AgriVision

> 👋 HI Guys , Hope you’ve already seen how our Machine Learning models work!  
If not, no worries — you can check it out here first:  
*Previous: [MACHINE_LEARNING.md](./MACHINE_LEARNING.md)*

🚀 Now that you’re all set, welcome to the **Deep Learning zone**!  
Let’s explore how CNN and LSTM models power **Plant Doctor** and **Price Forecasting**.

---

## 🎯 What is Deep Learning?

**Deep Learning (DL)** is a subset of Machine Learning that uses artificial neural networks inspired by the human brain.

```
┌───────────────────────────────────────────────┐
│               Machine Learning                │
│                                               │
│   ┌───────────────────────────────────────┐   │
│   │             Deep Learning             │   │
│   │                                       │   │
│   │   ┌───────────────────────────────┐   │   │
│   │   │   CNN (Image-based Models)    │   │   │
│   │   └───────────────────────────────┘   │   │
│   │                                       │   │
│   │   ┌───────────────────────────────┐   │   │
│   │   │  LSTM (Sequence / Time-Series)│   │   │
│   │   └───────────────────────────────┘   │   │
│   │                                       │   │
│   └───────────────────────────────────────┘   │
│                                               │
└───────────────────────────────────────────────┘

```

### ML vs DL

| Aspect | Machine Learning | Deep Learning |
|--------|------------------|---------------|
| **Data Type** | Tables, numbers | Images, text, audio |
| **Feature Engineering** | Manual (you create features) | Automatic (network learns) |
| **Training Time** | Minutes | Hours/Days |
| **Data Needed** | 1,000s of samples | 10,000s+ samples |
| **Hardware** | CPU is enough | GPU recommended |

---

## 🩺 1. Plant Doctor: CNN (Convolutional Neural Network)

### What it does
Takes a leaf image → Detects if the plant has a disease.

### How CNN "Sees" an Image

```
📷 Leaf Image (224x224 pixels)
        ↓
┌─────────────────────────────────────────────────────┐
│ Layer 1: EDGE DETECTION                             │
│ Finds simple patterns: lines, curves, edges         │
│ "I see a curved edge here"                          │
└─────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────┐
│ Layer 2: TEXTURE DETECTION                          │
│ Combines edges into textures                        │
│ "This area has a spotted texture"                   │
└─────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────┐
│ Layer 3: PATTERN DETECTION                          │
│ Recognizes complex patterns                         │
│ "This looks like disease spots"                     │
└─────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────┐
│ Layer 4: CLASSIFICATION                             │
│ Makes final decision                                │
│ "This is: Tomato Late Blight (95% confident)"       │
└─────────────────────────────────────────────────────┘
```

### Transfer Learning: MobileNetV2

Instead of training from scratch (needs millions of images), we use **Transfer Learning**:

```
┌─────────────────────────────────────────────────────┐
│ MobileNetV2 (Pre-trained on ImageNet)               │
│ Already knows: edges, textures, shapes, objects     │
│ Trained on: 14 million images, 1000 categories      │
└─────────────────────────────────────────────────────┘
        │
        │ We FREEZE these layers (keep the knowledge)
        ↓
┌─────────────────────────────────────────────────────┐
│ OUR CUSTOM LAYERS                                   │
│ Learn: "What do diseased leaves look like?"         │
│ Trained on: PlantVillage dataset (50,000 images)    │
└─────────────────────────────────────────────────────┘
        ↓
        🎯 Output: Disease Name + Confidence %
```

**Why MobileNetV2?**
- ✅ Lightweight (can run on mobile/edge devices)
- ✅ Fast inference (quick predictions)
- ✅ High accuracy with small datasets
- ✅ Pre-trained = less training needed

### CNN Architecture
```python
Input: 224 x 224 x 3 (RGB image)
    ↓
MobileNetV2 Base (frozen)
    ↓
Global Average Pooling
    ↓
Dense Layer (256 neurons) + ReLU
    ↓
Dropout (0.3) - prevents overfitting
    ↓
Dense Layer (15 neurons) + Softmax
    ↓
Output: 15 disease probabilities
```

---

## 💰 2. Market Prices: LSTM (Long Short-Term Memory)

### What it does
Takes historical prices → Predicts future prices (7 days).

### Why LSTM for Time Series?

Regular neural networks have no "memory"—they treat each input independently. But **prices depend on past prices**!

```
Traditional NN:  Price today = f(today's features)
LSTM:            Price today = f(today's features + yesterday + last week...)
```

### How LSTM "Remembers"

```
Day 1 Price: ₹2000 ──→ 🧠 LSTM Cell ──→ Remember: "Started at ₹2000"
                              │
Day 2 Price: ₹2050 ──→ 🧠 LSTM Cell ──→ Update: "Rising trend (+₹50)"
                              │
Day 3 Price: ₹2100 ──→ 🧠 LSTM Cell ──→ Update: "Still rising (+₹50/day)"
                              │
Day 4 Price: ₹2080 ──→ 🧠 LSTM Cell ──→ Update: "Slight correction"
                              │
                              ↓
                   📈 Predict Day 5-11
```

### LSTM Cell: The Memory Unit

Each LSTM cell has **3 gates**:

```
┌──────────────────────────────────────────────────────┐
│                    LSTM CELL                         │
│                                                      │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐           │
│  │ FORGET  │    │  INPUT  │    │ OUTPUT  │           │
│  │  GATE   │    │  GATE   │    │  GATE   │           │
│  └────┬────┘    └────┬────┘    └────┬────┘           │
│       │              │              │                │
│       ↓              ↓              ↓                │
│  "What to      "What new      "What to               │
│   forget?"      info to add?"   output?"             │
│                                                      │
│  Example:       Example:        Example:             │
│  Forget last    Add today's     Output the           │
│  year's data    price trend     prediction           │
└──────────────────────────────────────────────────────┘
```

### LSTM Architecture for Price Forecasting
```python
Input: 60 days of prices → Shape: (60, 1)
    ↓
LSTM Layer (50 units, return_sequences=True)
    ↓
LSTM Layer (50 units)
    ↓
Dense Layer (25 units)
    ↓
Dense Layer (7 units) ← 7-day forecast
    ↓
Output: [Day1, Day2, Day3, Day4, Day5, Day6, Day7] prices
```

---

## 🔧 Key Deep Learning Concepts

### Activation Functions

```
ReLU (Rectified Linear Unit)
─────────────────────────────
    │    ╱
    │   ╱
    │  ╱
────┼─╱──────→  If x > 0: output x
    │           If x < 0: output 0

Softmax (for classification)
─────────────────────────────
Input: [2.0, 1.0, 0.5]
Output: [0.65, 0.24, 0.11]  ← probabilities that sum to 1
```

### Loss Functions

| Task | Loss Function | What it measures |
|------|---------------|------------------|
| Classification (Disease) | CrossEntropy | How wrong the probability is |
| Regression (Price) | MSE | Average squared error |

### Optimizer: Adam

Adam automatically adjusts learning speed:
- 🐢 Slow down when close to the answer
- 🐇 Speed up when far from the answer

### Epochs & Batches

```
Dataset: 10,000 images
Batch Size: 32

1 Epoch = Process all 10,000 images once
        = 10,000 / 32 = 313 batches

Training: 30 epochs = See each image 30 times
```

---

## 📊 Model Training Flow

```
            ┌─────────────────────┐
            │   Load Dataset      │
            │   (Images/Prices)   │
            └──────────┬──────────┘
                       ↓
            ┌─────────────────────┐
            │   Preprocess        │
            │   • Resize images   │
            │   • Normalize (0-1) │
            │   • Augment data    │
            └──────────┬──────────┘
                       ↓
            ┌─────────────────────┐
            │   Build Model       │
            │   • Define layers   │
            │   • Set activations │
            └──────────┬──────────┘
                       ↓
            ┌─────────────────────┐
            │   Compile Model     │
            │   • Loss function   │
            │   • Optimizer       │
            │   • Metrics         │
            └──────────┬──────────┘
                       ↓
            ┌─────────────────────┐
            │   Train (fit)       │◄──────────┐
            │   • Forward pass    │           │
            │   • Calculate loss  │           │
            │   • Backpropagate   │  Repeat   │
            │   • Update weights  │  (epochs) │
            └──────────┬──────────┘───────────┘
                       ↓
            ┌─────────────────────┐
            │   Evaluate          │
            │   • Accuracy        │
            │   • Loss curves     │
            └──────────┬──────────┘
                       ↓
            ┌─────────────────────┐
            │   Save Model        │
            │   (.h5 file)        │
            └─────────────────────┘
```

---

## 💡 Summary Comparison

```
┌────────────────────────────────────────────────────────────────┐
│                      DEEP LEARNING                             │
│                                                                │
│   ┌─────────────────────┐    ┌─────────────────────┐           │
│   │        CNN          │    │        LSTM         │           │
│   │   (Plant Doctor)    │    │   (Price Forecast)  │           │
│   ├─────────────────────┤    ├─────────────────────┤           │
│   │ Input: Images       │    │ Input: Time Series  │           │
│   │ Learns: Patterns    │    │ Learns: Sequences   │           │
│   │ Output: Class       │    │ Output: Future vals │           │
│   │                     │    │                     │           │
│   │ Used for:           │    │ Used for:           │           │
│   │ • Image classify    │    │ • Stock prediction  │           │
│   │ • Object detection  │    │ • Weather forecast  │           │
│   │ • Face recognition  │    │ • Language models   │           │
│   └─────────────────────┘    └─────────────────────┘           │
│                                                                │
│   Key Difference from ML:                                      │
│   • Automatically learns features (no manual engineering)      │
│   • Needs MORE data                                            │
│   • Needs MORE compute (GPU recommended)                       │
│   • Better for unstructured data (images, audio, text)         │
│                                                                │
│   AgriVision uses: MobileNetV2 (CNN), LSTM                     │
└────────────────────────────────────────────────────────────────┘
```

---

## 🔗 Quick Reference

| What | ML (XGBoost/RF) | DL (CNN/LSTM) |
|------|-----------------|---------------|
| Yield Prediction | ✅ Best choice | Overkill |
| Crop Recommendation | ✅ Best choice | Overkill |
| Disease Detection | ❌ Can't handle images | ✅ Best choice |
| Price Forecasting | ⚠️ Works okay | ✅ Better for trends |

---
