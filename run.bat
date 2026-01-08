@echo off
echo Installing requirements...
pip install -r requirements.txt

echo Training Models...
python train_models.py

echo Starting Application...
python app.py
pause
