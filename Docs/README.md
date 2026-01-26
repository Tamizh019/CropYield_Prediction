# 📚 AgriVision Documentation

Welcome to the technical documentation for AgriVision v3.1

---

## 📖 Quick Navigation

### Getting Started
| Doc | Description |
|-----|-------------|
| [SETUP.md](./SETUP.md) | Installation, dependencies, environment |
| [API_REFERENCE.md](./API_REFERENCE.md) | Flask routes & endpoints |

### System Design
| Doc | Description |
|-----|-------------|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | System diagram, data flow |
| [MODELS_OVERVIEW.md](./MODELS_OVERVIEW.md) | All ML/DL model specs |

### Deep Dives
| Doc | Description |
|-----|-------------|
| [MACHINE_LEARNING.md](./MACHINE_LEARNING.md) | XGBoost, Random Forest details |
| [DEEP_LEARNING.md](./DEEP_LEARNING.md) | CNN, TensorFlow |
| [CNN_Training_Explained.md](./CNN_Training_Explained.md) | Plant disease model training |

### History
| Doc | Description |
|-----|-------------|
| [CHANGELOG.md](./CHANGELOG.md) | Version history (v1.0 → v3.1) |

---

## 🆕 Latest Updates (v3.1)

### ML Analytics Dashboard
The bulk prediction result page now includes:
- **Model Confidence**: Based on prediction variance analysis
- **Feature Importance**: Extracted from XGBoost model
- **Yield Distribution**: Histogram showing prediction ranges
- **Prediction Classification**: High/Medium/Low yield counts

### AI Farming Advisor
Replaced generic data descriptions with actionable recommendations:
- Priority Actions
- Yield Improvement Strategies
- Risk Mitigation
- Growth Opportunities

### Key Bug Fixes
- State names now display correctly (was showing numeric codes)
- Improved page spacing for better readability

---

## 🗂️ File Index

```
Docs/
├── README.md              # This file
├── SETUP.md               # Installation guide
├── API_REFERENCE.md       # Route documentation
├── ARCHITECTURE.md        # System design
├── MODELS_OVERVIEW.md     # Model specifications
├── MACHINE_LEARNING.md    # ML algorithms
├── DEEP_LEARNING.md       # Neural networks
├── CNN_Training_Explained.md  # CNN details
└── CHANGELOG.md           # Version history
```

---

*Team AgriVision*
