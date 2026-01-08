# Crop Yield Prediction System 🌾
Hey Guys! This is our machine learning project (Example) that helps farmers decide what to grow and how much they can expect to harvest.

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

### Running the App
It's super easy. We made a script for it.
1. Just double-click the file named **`run.bat`**.
2. It will install everything and start the website for you.
3. Open `http://127.0.0.1:5000` in your browser.

### OR (Manual Way)
If you want to run manually, open CMD in this folder and run:
1. `pip install -r requirements.txt`
2. `python train_models.py`
3. `python app.py`

## Technologies Used
- **Python & Flask:** For the backend server.
- **Google Gemini AI:** For smart agronomy insights.
- **Machine Learning:** XGBoost & Ensemble Classifiers (High Accuracy).
- **Glassmorphism UI:** Modern and beautiful design.

## Note 📢
Guys , If you can't train the model for any reason, you can download the trained models from my drive:
👉 [Download Models Here](https://drive.google.com/drive/folders/1gMGjGMz0oCBkrMp2QtCyx14zPZ9lk4Y5?usp=sharing)

I already trained it for you! Just put the files in the `models/` folder.

Enjoy! 🚀
