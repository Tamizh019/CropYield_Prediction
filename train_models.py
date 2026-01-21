import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, mean_absolute_error, classification_report
from xgboost import XGBRegressor, XGBClassifier
import joblib
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Create directories
if not os.path.exists('models'):
    os.makedirs('models')

if not os.path.exists('logs'):
    os.makedirs('logs')

# Setup logging
log_file = f'logs/training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
log = open(log_file, 'w', encoding='utf-8')

def print_log(message):
    """Print and log simultaneously"""
    print(message)
    try:
        log.write(message + '\n')
        log.flush()
    except UnicodeEncodeError:
        # Fallback: Remove emojis if encoding fails
        log.write(message.encode('ascii', 'ignore').decode('ascii') + '\n')
        log.flush()

# ========================================
# YIELD PREDICTION MODEL (ADVANCED)
# ========================================

def train_yield_model():
    """
    Train an advanced yield prediction model using ensemble methods
    """
    print_log("\n" + "="*60)
    print_log("🌾 TRAINING YIELD PREDICTION MODEL (ADVANCED)")
    print_log("="*60)
    
    try:
        df = pd.read_csv('Datasets/Crop_yield_india.csv')
        print_log(f"✅ Dataset loaded: {len(df)} rows")
    except Exception as e:
        print_log(f"❌ Failed to read Yield CSV: {e}")
        return
    
    # Clean column names
    df.columns = df.columns.str.strip()
    print_log(f"📊 Columns: {df.columns.tolist()}")
    
    # Identify target column
    target_col = None
    if 'Yield_kg_per_ha' in df.columns:
        target_col = 'Yield_kg_per_ha'
    elif 'Yield' in df.columns:
        target_col = 'Yield'
    elif 'Production' in df.columns and 'Area' in df.columns:
        df = df[df['Area'] > 0]
        df['Yield'] = df['Production'] / df['Area']
        target_col = 'Yield'
    else:
        print_log("❌ ERROR: Could not identify target column")
        return
    
    print_log(f"🎯 Target Column: {target_col}")
    
    # Data Cleaning
    initial_rows = len(df)
    df = df.dropna()
    df = df[df[target_col] > 0]  # Remove invalid yields
    print_log(f"🧹 Cleaned data: {len(df)} rows (removed {initial_rows - len(df)})")
    
    # Feature mapping
    column_mapping = {
        'State Name': 'State_Name',
        'Dist Name': 'District_Name',
        'Year': 'Crop_Year',
        'Crop': 'Crop',
        'Area_ha': 'Area',
        'Temperature_C': 'Temperature',
        'Humidity_%': 'Humidity',
        'pH': 'pH',
        'Rainfall_mm': 'Rainfall'
    }
    
    # Create standardized features
    selected_features = []
    for csv_col, std_col in column_mapping.items():
        if csv_col in df.columns:
            df[std_col] = df[csv_col]
            selected_features.append(std_col)
        else:
            print_log(f"⚠️ Feature '{csv_col}' not found")
    
    if not selected_features:
        print_log("❌ No valid features found")
        return
    
    print_log(f"✅ Features selected: {selected_features}")
    
    # Feature Engineering
    print_log("\n🔧 Feature Engineering...")
    
    # Create interaction features
    if 'Temperature' in df.columns and 'Humidity' in df.columns:
        df['Temp_Humidity_Interaction'] = df['Temperature'] * df['Humidity']
        selected_features.append('Temp_Humidity_Interaction')
    
    if 'pH' in df.columns and 'Rainfall' in df.columns:
        df['pH_Rainfall_Interaction'] = df['pH'] * df['Rainfall']
        selected_features.append('pH_Rainfall_Interaction')
    
    # Polynomial features for environmental factors
    if 'Temperature' in df.columns:
        df['Temperature_Squared'] = df['Temperature'] ** 2
        selected_features.append('Temperature_Squared')
    
    if 'Rainfall' in df.columns:
        df['Rainfall_Log'] = np.log1p(df['Rainfall'])
        selected_features.append('Rainfall_Log')
    
    print_log(f"✅ Total features after engineering: {len(selected_features)}")
    
    # Prepare data
    X = df[selected_features].copy()
    y = df[target_col]
    
    # Encode categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    print_log(f"🔤 Encoding {len(categorical_cols)} categorical columns...")
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print_log(f"   - {col}: {len(le.classes_)} unique values")
    
    # Save encoders
    joblib.dump(label_encoders, 'models/yield_label_encoders.pkl')
    print_log("💾 Label encoders saved")
    
    # Scale numerical features (optional but improves XGBoost)
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    scaler = RobustScaler()  # Robust to outliers
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    joblib.dump(scaler, 'models/yield_scaler.pkl')
    print_log("💾 Scaler saved")
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print_log(f"\n📊 Training set: {len(X_train)} | Test set: {len(X_test)}")
    
    # ========================================
    # MODEL TRAINING - ENSEMBLE APPROACH
    # ========================================
    
    print_log("\n🤖 Training Multiple Models...")
    
    models = {}
    scores = {}
    
    # 1. Random Forest (Baseline)
    print_log("\n1️⃣ Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_mae = mean_absolute_error(y_test, rf_pred)
    
    models['RandomForest'] = rf_model
    scores['RandomForest'] = {'R2': rf_r2, 'RMSE': rf_rmse, 'MAE': rf_mae}
    
    print_log(f"   R² Score: {rf_r2:.4f}")
    print_log(f"   RMSE: {rf_rmse:.2f}")
    print_log(f"   MAE: {rf_mae:.2f}")
    
    # 2. Gradient Boosting
    print_log("\n2️⃣ Gradient Boosting Regressor...")
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=7,
        min_samples_split=5,
        random_state=42,
        verbose=0
    )
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_r2 = r2_score(y_test, gb_pred)
    gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
    gb_mae = mean_absolute_error(y_test, gb_pred)
    
    models['GradientBoosting'] = gb_model
    scores['GradientBoosting'] = {'R2': gb_r2, 'RMSE': gb_rmse, 'MAE': gb_mae}
    
    print_log(f"   R² Score: {gb_r2:.4f}")
    print_log(f"   RMSE: {gb_rmse:.2f}")
    print_log(f"   MAE: {gb_mae:.2f}")
    
    # 3. XGBoost (Best performer)
    print_log("\n3️⃣ XGBoost Regressor...")
    xgb_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=3,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_r2 = r2_score(y_test, xgb_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    
    models['XGBoost'] = xgb_model
    scores['XGBoost'] = {'R2': xgb_r2, 'RMSE': xgb_rmse, 'MAE': xgb_mae}
    
    print_log(f"   R² Score: {xgb_r2:.4f}")
    print_log(f"   RMSE: {xgb_rmse:.2f}")
    print_log(f"   MAE: {xgb_mae:.2f}")
    
    # ========================================
    # SELECT BEST MODEL
    # ========================================
    
    print_log("\n" + "="*60)
    print_log("📊 MODEL COMPARISON")
    print_log("="*60)
    
    best_model_name = None
    best_r2 = -np.inf
    
    for name, score in scores.items():
        print_log(f"{name:20s} | R²: {score['R2']:.4f} | RMSE: {score['RMSE']:8.2f} | MAE: {score['MAE']:8.2f}")
        if score['R2'] > best_r2:
            best_r2 = score['R2']
            best_model_name = name
    
    best_model = models[best_model_name]
    
    print_log(f"\n🏆 BEST MODEL: {best_model_name} (R² = {best_r2:.4f})")
    
    # Save best model
    joblib.dump(best_model, 'models/yield_model.pkl')
    joblib.dump(selected_features, 'models/yield_features.pkl')
    
    # Save model metadata
    metadata = {
        'model_type': best_model_name,
        'r2_score': best_r2,
        'rmse': scores[best_model_name]['RMSE'],
        'mae': scores[best_model_name]['MAE'],
        'features': selected_features,
        'trained_on': datetime.now().isoformat(),
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    joblib.dump(metadata, 'models/yield_metadata.pkl')
    
    print_log("\n✅ Yield Model Training Complete!")
    print_log(f"💾 Models saved in 'models/' directory")


# ========================================
# CROP RECOMMENDATION MODEL (ADVANCED)
# ========================================

def train_recommendation_model():
    """
    Train an advanced crop recommendation model
    """
    print_log("\n" + "="*60)
    print_log("🌾 TRAINING CROP RECOMMENDATION MODEL (ADVANCED)")
    print_log("="*60)
    
    try:
        df = pd.read_csv('Datasets/Crop_recommendation.csv')
        print_log(f"✅ Dataset loaded: {len(df)} rows")
    except Exception as e:
        print_log(f"❌ Failed to read Recommendation CSV: {e}")
        return
    
    # Clean column names
    df.columns = df.columns.str.strip()
    print_log(f"📊 Columns: {df.columns.tolist()}")
    
    # Identify target column
    target_col = 'label'
    if target_col not in df.columns:
        obj_cols = df.select_dtypes(include=['object']).columns
        if len(obj_cols) == 1:
            target_col = obj_cols[0]
            print_log(f"🎯 Assumed target column: {target_col}")
        else:
            print_log("❌ Could not identify target column")
            return
    
    # Data preparation
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    print_log(f"🎯 Target classes: {y.nunique()} unique crops")
    print_log(f"📊 Class distribution:\n{y.value_counts().head(10)}")

    # Encode target labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    joblib.dump(le, 'models/recommend_label_encoder.pkl')
    print_log("💾 Target label encoder saved")
    
    # Feature Engineering
    print_log("\n🔧 Feature Engineering...")
    
    # NPK ratio features
    if all(col in X.columns for col in ['N', 'P', 'K']):
        X['NPK_Sum'] = X['N'] + X['P'] + X['K']
        X['NPK_Ratio'] = X['N'] / (X['P'] + X['K'] + 1)
        X['NK_Ratio'] = X['N'] / (X['K'] + 1)
        X['PK_Ratio'] = X['P'] / (X['K'] + 1)
        print_log("   ✅ NPK ratio features created")
    
    # Environmental interaction features
    if 'temperature' in X.columns and 'humidity' in X.columns:
        X['temp_humidity'] = X['temperature'] * X['humidity']
        print_log("   ✅ Temperature-Humidity interaction created")
    
    if 'rainfall' in X.columns and 'ph' in X.columns:
        X['rainfall_ph'] = X['rainfall'] * X['ph']
        print_log("   ✅ Rainfall-pH interaction created")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    joblib.dump(scaler, 'models/recommend_scaler.pkl')
    print_log("💾 Scaler saved")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print_log(f"\n📊 Training set: {len(X_train)} | Test set: {len(X_test)}")
    
    # ========================================
    # MODEL TRAINING
    # ========================================
    
    print_log("\n🤖 Training Multiple Models...")
    
    models = {}
    scores = {}
    
    # 1. Random Forest
    print_log("\n1️⃣ Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    
    models['RandomForest'] = rf_model
    scores['RandomForest'] = rf_acc
    
    print_log(f"   Accuracy: {rf_acc:.4f}")
    
    # 2. Gradient Boosting
    print_log("\n2️⃣ Gradient Boosting Classifier...")
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=0
    )
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_acc = accuracy_score(y_test, gb_pred)
    
    models['GradientBoosting'] = gb_model
    scores['GradientBoosting'] = gb_acc
    
    print_log(f"   Accuracy: {gb_acc:.4f}")
    
    # 3. XGBoost
    print_log("\n3️⃣ XGBoost Classifier...")
    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    
    models['XGBoost'] = xgb_model
    scores['XGBoost'] = xgb_acc
    
    print_log(f"   Accuracy: {xgb_acc:.4f}")
    
    # ========================================
    # SELECT BEST MODEL
    # ========================================
    
    print_log("\n" + "="*60)
    print_log("📊 MODEL COMPARISON")
    print_log("="*60)
    
    best_model_name = max(scores, key=scores.get)
    best_model = models[best_model_name]
    best_acc = scores[best_model_name]
    
    for name, acc in scores.items():
        print_log(f"{name:20s} | Accuracy: {acc:.4f}")
    
    print_log(f"\n🏆 BEST MODEL: {best_model_name} (Accuracy = {best_acc:.4f})")
    
    # Detailed classification report
    best_pred = best_model.predict(X_test)
    print_log("\n📋 Classification Report:")
    print_log(classification_report(y_test, best_pred))
    
    # Save best model
    joblib.dump(best_model, 'models/recommend_model.pkl')
    
    # Save metadata
    metadata = {
        'model_type': best_model_name,
        'accuracy': best_acc,
        'num_classes': len(list(set(y))),
        'features': X.columns.tolist(),
        'trained_on': datetime.now().isoformat(),
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    joblib.dump(metadata, 'models/recommend_metadata.pkl')
    
    print_log("\n✅ Crop Recommendation Model Training Complete!")
    print_log(f"💾 Model saved in 'models/' directory")


# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == '__main__':
    start_time = datetime.now()
    
    print_log("="*60)
    print_log("🌾 AGRIVISION MODEL TRAINING PIPELINE")
    print_log(f"⏰ Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print_log("="*60)
    
    try:
        # Train both models
        train_yield_model()
        train_recommendation_model()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print_log("\n" + "="*60)
        print_log("✅ ALL MODELS TRAINED SUCCESSFULLY!")
        print_log(f"⏱️ Total Duration: {duration:.2f} seconds")
        print_log(f"📁 Log saved to: {log_file}")
        print_log("="*60)
        
    except Exception as e:
        print_log(f"\n❌ TRAINING FAILED: {e}")
        import traceback
        print_log(traceback.format_exc())
    
    finally:
        log.close()
