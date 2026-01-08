# Crop Yield Prediction System 🌾

Hey Guys! This is our machine learning project (example) that helps farmers decide what to grow and how much they can expect to harvest.

## Team Members 👥
- **Arunmozhi Adithya**
- **Jenivaa**
- **Pradeepraja**
- **Dilshan**

## What does it do?
It uses AI to look at things like soil quality, rain, and temperature to tell you two main things:
1. **Yield Prediction:** If you plant a crop, how much of it will you get? (in tonnes).
2. **Crop Recommendation:** Based on your soil, what is the best crop to grow?

## Features
- **Smart Forms:** Just enter the details (like State, District, nitrogen levels, etc.) and get an instant answer.
- **Bulk Upload:** Got a big excel/csv file? Upload it and get predictions for hundreds of rows at once.
- **Dashboard:** See cool charts and stats about the data you uploaded (like which state performs best).

## How to run it?
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
- **HTML/CSS:** For the website design (it looks pretty modern!).
- **Machine Learning:** Random Forest algorithm (it's accurate!).

## Note 📢
Guys, if you can't train the model for any reason, you can download the already trained models from my drive:
👉 [Download Models Here](https://drive.google.com/drive/folders/1gMGjGMz0oCBkrMp2QtCyx14zPZ9lk4Y5?usp=sharing)

I already trained it for you! Just put the files in the `models/` folder.

Enjoy! 🚀
