"""
YieldMax Ensemble Training Script
Train the unified ensemble model (XGBoost + LightGBM + DNN + Stacking)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from ensemble_model import YieldMaxEnsemble
from datetime import datetime

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

print("="*60)
print("🌾 YieldMax Precision Model - Training Pipeline")
print("="*60)

# ========================================
# LOAD & PREPROCESS DATA
# ========================================

print("\n[1/5] Loading dataset...")
dataset_path = 'Datasets/Yield_Data_Ready.csv'

if not os.path.exists(dataset_path):
    print(f"❌ Dataset not found at {dataset_path}")
    print("   Please run 'python scripts/prepare_dataset.py' first!")
    sys.exit(1)

df = pd.read_csv(dataset_path)

# Rename Year to Crop_Year if needed
if 'Year' in df.columns and 'Crop_Year' not in df.columns:
    df.rename(columns={'Year': 'Crop_Year'}, inplace=True)
    print("   Renamed 'Year' to 'Crop_Year'")

print(f"✅ Loaded {len(df)} records")

# ========================================
# COMPUTE YIELD (TARGET)
# ========================================

print("\n[1.5/5] Computing Yield (T/Ha) = Production / Area...")

df = df[df['Area'] > 0].copy()
df['Yield'] = df['Production'] / df['Area']

before = len(df)
df = df[(df['Yield'] >= 0.01) & (df['Yield'] <= 50)].copy()
after = len(df)
print(f"   Clipped {before - after} outlier rows (Yield outside 0.01-50 T/Ha range)")
print(f"   Remaining records: {after}")
print(f"   Yield range: {df['Yield'].min():.3f} - {df['Yield'].max():.3f} T/Ha")
print(f"   Avg Yield: {df['Yield'].mean():.3f} T/Ha")

# ========================================
# FEATURE ENGINEERING
# ========================================

print("\n[2/5] Feature engineering...")

# Encode categorical features (NO SEASON - only what exists in dataset)
categorical_cols = ['State_Name', 'District_Name', 'Crop']
label_encoders = {}

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"   Encoded {col}: {len(le.classes_)} unique values")

# Define features and target (9 features - NO SEASON)
feature_cols = ['State_Name', 'District_Name', 'Crop_Year', 'Crop', 'Area',
                'Temperature', 'Humidity', 'pH', 'Rainfall']

# Check for missing columns
missing_cols = [col for col in feature_cols if col not in df.columns]
if missing_cols:
    print(f"❌ Missing required columns: {missing_cols}")
    sys.exit(1)

X = df[feature_cols].copy()
y = df['Yield'].values  # Target: Yield in T/Ha (Production / Area)

print(f"✅ Features: {X.shape[1]} columns")
print(f"✅ Target: Yield T/Ha (min={y.min():.3f}, max={y.max():.3f}, avg={y.mean():.3f})")

# ========================================
# TRAIN-TEST SPLIT
# ========================================

print("\n[3/5] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Training set: {len(X_train)} samples")
print(f"   Test set: {len(X_test)} samples")

# ========================================
# TRAIN YIELDMAX ENSEMBLE
# ========================================

print("\n[4/5] Training YieldMax Ensemble...")
print("   This will train 3 models + meta-learner...")
print("   ⏳ Please wait (this may take several minutes)...\n")

ensemble = YieldMaxEnsemble()

try:
    metrics = ensemble.train(X_train.values, y_train, feature_names=feature_cols)
    
    print("\n✅ Training Complete!")
    print("\n" + "="*60)
    print("📊 TRAINING SUMMARY")
    print("="*60)
    print(f"   Status: {metrics.get('status', 'unknown')}")
    print(f"   Features: {metrics.get('n_features', 'N/A')}")
    print(f"   Training Samples: {metrics.get('n_samples', 'N/A')}")
    print("="*60)
    
except Exception as e:
    print(f"❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ========================================
# EVALUATE ON TEST SET
# ========================================

print("\n[5/5] Evaluating on test set...")

try:
    # Get predictions
    result = ensemble.predict(X_test.values, return_details=True)
    predictions = result['final_prediction']
    confidence = result['confidence']
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\n" + "="*60)
    print("🎯 TEST SET PERFORMANCE (YieldMax Ensemble)")
    print("="*60)
    print(f"   R² Score: {r2:.4f}")
    print(f"   RMSE: {rmse:.2f} tonnes")
    print(f"   MAE: {mae:.2f} tonnes")
    print(f"   Avg Confidence: {np.mean(confidence):.1f}%")
    print("="*60)
    
    # Sample predictions
    print("\n📋 Sample Predictions:")
    print("-" * 60)
    print(f"{'Actual':<15} {'Predicted':<15} {'Confidence':<15} {'Error'}")
    print("-" * 60)
    for i in range(min(10, len(predictions))):
        actual = y_test[i]
        pred = predictions[i]
        conf = confidence[i]
        error = abs(actual - pred)
        print(f"{actual:<15.2f} {pred:<15.2f} {conf:<15.1f}% {error:.2f}")
    print("-" * 60)
    
except Exception as e:
    print(f"⚠️ Evaluation error: {e}")

# ========================================
# SAVE MODELS
# ========================================

print("\n[6/6] Saving models...")

try:
    # Save YieldMax Ensemble
    ensemble_path = 'models/yieldmax_ensemble.pkl'
    ensemble.save(ensemble_path)
    print(f"✅ Saved: {ensemble_path}")
    
    # Save label encoders
    joblib.dump(label_encoders, 'models/yield_label_encoders.pkl')
    print("✅ Saved: models/yield_label_encoders.pkl")
    
    # Save feature names
    joblib.dump(feature_cols, 'models/yield_features.pkl')
    print("✅ Saved: models/yield_features.pkl")
    
    # Save metadata
    metadata = {
        'model_name': 'YieldMax Precision Model',
        'model_type': 'Ensemble (XGBoost + LightGBM + DNN + Stacking)',
        'training_date': datetime.now().isoformat(),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'r2_score': r2,
        'rmse': rmse,
        'mae': mae,
        'features': feature_cols,
        'num_features': len(feature_cols)
    }
    joblib.dump(metadata, 'models/ensemble_metadata.pkl')
    print("✅ Saved: models/ensemble_metadata.pkl")
    
    # Save training log
    log_filename = f"logs/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_filename, 'w') as f:
        f.write("="*60 + "\n")
        f.write("YieldMax Precision Model - Training Log\n")
        f.write("="*60 + "\n\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Total Records: {len(df)}\n")
        f.write(f"Training Set: {len(X_train)}\n")
        f.write(f"Test Set: {len(X_test)}\n\n")
        
        f.write("="*60 + "\n")
        f.write("ENSEMBLE PERFORMANCE\n")
        f.write("="*60 + "\n")
        f.write(f"R² Score: {r2:.4f}\n")
        f.write(f"RMSE: {rmse:.2f} tonnes\n")
        f.write(f"MAE: {mae:.2f} tonnes\n")
        f.write(f"Avg Confidence: {np.mean(confidence):.1f}%\n\n")
        
        f.write("="*60 + "\n")
        f.write("INDIVIDUAL MODEL METRICS\n")
        f.write("="*60 + "\n")
        for model_name, model_metrics in metrics.items():
            f.write(f"\n{model_name.upper()}:\n")
            if not isinstance(model_metrics, dict):
                f.write(f"   {model_metrics}\n")
                continue
            for metric, value in model_metrics.items():
                if isinstance(value, float):
                    f.write(f"   {metric}: {value:.4f}\n")
                else:
                    f.write(f"   {metric}: {value}\n")
    
    print(f"✅ Saved: {log_filename}")
    
except Exception as e:
    print(f"⚠️ Save error: {e}")

print("\n" + "="*60)
print("🎉 TRAINING COMPLETE!")
print("="*60)
print("\nYour YieldMax Precision Model is ready to use.")
print("\nNext steps:")
print("  1. Run the Flask app: python app.py")
print("  2. Navigate to http://localhost:5000")
print("  3. Try the YieldMax Predictor!")
print("\n" + "="*60 + "\n")
