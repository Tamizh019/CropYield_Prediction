"""
Crop Recommendation Model Retraining Script
Trains with balanced dataset + consistent 13-feature engineering
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime

print("=" * 60)
print("Crop Recommendation Model - Retraining Pipeline")
print("=" * 60)

# Load Dataset
df = pd.read_csv('Datasets/Crop_recommendation.csv')
print(f"Loaded {len(df)} records")
print(f"Columns: {list(df.columns)}")
print(f"Crops: {df['label'].nunique()} unique classes")
print(f"Class distribution:\n{df['label'].value_counts().to_string()}")

# Feature Engineering - SAME as app.py inference code (13 features)
print("\n[2/5] Feature engineering (13 features)...")

X_raw = df[['N','P','K','temperature','humidity','ph','rainfall']].copy()
X_raw['NPK_Sum']       = X_raw['N'] + X_raw['P'] + X_raw['K']
X_raw['NPK_Ratio']     = X_raw['N'] / (X_raw['P'] + X_raw['K'] + 1)
X_raw['NK_Ratio']      = X_raw['N'] / (X_raw['K'] + 1)
X_raw['PK_Ratio']      = X_raw['P'] / (X_raw['K'] + 1)
X_raw['temp_humidity'] = X_raw['temperature'] * X_raw['humidity']
X_raw['rainfall_ph']   = X_raw['rainfall'] * X_raw['ph']

FEATURE_COLS = ['N','P','K','temperature','humidity','ph','rainfall',
                'NPK_Sum','NPK_Ratio','NK_Ratio','PK_Ratio','temp_humidity','rainfall_ph']

X = X_raw[FEATURE_COLS].values
y = df['label'].values
print(f"Feature matrix: {X.shape}")

# Encode Labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"Classes: {list(le.classes_)}")

# Train/Test Split (stratified)
print("\n[3/5] Splitting data (stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# Scale Features
print("\n[4/5] Scaling & training RandomForestClassifier...")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Train Model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_sc, y_train)

# Evaluate
y_pred = model.predict(X_test_sc)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_sc, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"CV Accuracy: {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")

# Sanity Check Predictions
print("\n[Sanity] Testing with known conditions...")
tests = [
    ('Rice (high N, rain)',    [90, 42, 43, 22.0, 82.0, 6.5, 230.0]),
    ('Maize (low rain)',       [80, 45, 20, 22.0, 65.0, 6.5,  90.0]),
    ('Chickpea (low hum)',     [40, 72, 80, 18.0, 17.0, 7.2,  80.0]),
    ('Wheat (cool, med rain)', [70, 50, 40, 15.0, 55.0, 7.0, 400.0]),
    ('Cotton (hot, low K)',   [120, 36, 43, 32.0, 55.0, 6.5, 600.0]),
    ('Coffee (acidic, rainy)',[103, 18, 30, 25.0, 80.0, 6.8,1400.0]),
]
for name, vals in tests:
    N, P, K, temp, hum, ph, rain = vals
    feat = np.array([[N, P, K, temp, hum, ph, rain,
                      N+P+K, N/(P+K+1), N/(K+1), P/(K+1), temp*hum, rain*ph]])
    feat_sc = scaler.transform(feat)
    probs = model.predict_proba(feat_sc)[0]
    top3 = np.argsort(probs)[::-1][:3]
    results = [(le.classes_[i], round(probs[i]*100, 1)) for i in top3]
    print(f"  {name:30s} -> {results[0][0]} ({results[0][1]}%) | {results[1][0]} ({results[1][1]}%)")

# Save Models
print("\n[5/5] Saving models...")
os.makedirs('models', exist_ok=True)
joblib.dump(model,  'models/recommend_model.pkl')
joblib.dump(scaler, 'models/recommend_scaler.pkl')

metadata = {
    'model_name':    'Crop Recommendation RF',
    'model_type':    'RandomForestClassifier (balanced, 300 trees)',
    'training_date': datetime.now().isoformat(),
    'n_features':    len(FEATURE_COLS),
    'feature_cols':  FEATURE_COLS,
    'n_classes':     len(le.classes_),
    'classes':       list(le.classes_),
    'accuracy':      round(acc, 4),
    'cv_mean':       round(cv_scores.mean(), 4),
    'n_samples':     len(df),
}
joblib.dump(metadata, 'models/recommend_metadata.pkl')

print("Saved: models/recommend_model.pkl")
print("Saved: models/recommend_scaler.pkl")
print("Saved: models/recommend_metadata.pkl")
print("\n" + "=" * 60)
print(f"RETRAINING COMPLETE! Accuracy: {acc*100:.2f}%")
print("The recommend model now uses 13 consistent features.")
print("=" * 60)
