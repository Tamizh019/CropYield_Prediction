# 📜 Changelog

All notable changes to AgriVision, organized by version.

---

## [v3.1] - 2026-01-27 🎯

### Added
- **📊 ML Analytics Dashboard**: Professional analytics for bulk predictions
  - Model confidence score (60-95% based on prediction variance)
  - Feature importance chart (extracted from XGBoost model)
  - Yield distribution histogram (0-1000, 1000-2000, etc.)
  - Prediction classification (High/Medium/Low yield counts)
- **🤖 AI Farming Advisor**: Actionable recommendations replacing data descriptions
  - Priority Actions section
  - Yield Improvement Strategies
  - Risk Mitigation suggestions
  - Growth Opportunities analysis
- **📈 Enhanced Charts**: 4 interactive charts (was 2)
  - Horizontal bar chart for regional performance
  - Dual-axis chart for crop analysis (count + yield)

### Fixed
- **State name bug**: Now correctly shows "Chhattisgarh" instead of "10"
- **Condensed layout**: Increased spacing throughout bulk result page

### Changed
- Reduced data preview from 25 rows to 10 rows
- Improved AI prompt to focus on recommendations, not descriptions
- Enhanced chart styling with better colors and padding

---

## [v3.0] - 2026-01-20 🚀

### Added
- **🩺 Plant Doctor**: CNN-based disease detection (MobileNetV2)
- **💰 Market Prices**: LSTM price forecasting with real datasets
- **🌦️ Weather Service**: Live weather with 5-day forecast
- **🧪 Fertilizer Calculator**: NPK optimization recommendations
- **📊 Dashboard Status**: Real-time model status indicators

### Changed
- Upgraded to Gemini 2.0 Flash Experimental
- Glassmorphism UI design
- Premium HTML card-based AI insights

---

## [v2.0] - 2026-01-17

### Added
- **🌾 Yield Prediction**: XGBoost-based estimation
- **🧪 Crop Recommendation**: Random Forest classifier
- **📁 Bulk CSV Upload**: Process multiple predictions
- **🤖 AI Insights**: Gemini AI integration

### Changed
- Modern dark theme dashboard
- Chart.js visualizations

---

## [v1.0] - 2026-01-10

### Added
- Initial Flask application
- Basic yield prediction form
- Simple crop recommendation

---

## 🗺️ Roadmap

- [ ] PDF report download with charts
- [ ] Mobile app (React Native)
- [ ] Multi-language (Tamil, Hindi)
- [ ] Satellite imagery integration
- [ ] IoT sensor data support
