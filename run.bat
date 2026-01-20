@echo off
echo Installing requirements...
pip install -r requirements.txt

echo Training Models...
py train_models.py

echo Starting Application...
py app.py
pause
