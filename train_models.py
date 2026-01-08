import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
import joblib
import os
import sys

# Redirect stdout to a file if running in background without console access
# sys.stdout = open('model_training_log.txt', 'w')
# sys.stderr = sys.stdout

if not os.path.exists('models'):
    os.makedirs('models')

def train_yield_model():
    print("--- Training Yield Prediction Model ---")
    try:
        df = pd.read_csv('Datasets/Crop_yield_india.csv')
    except Exception as e:
        print(f"Failed to read Yield CSV: {e}")
        return

    print(f"Columns found: {df.columns.tolist()}")
    
    # Strip whitespace from columns
    df.columns = df.columns.str.strip()
    
    # Identify Target
    target_col = None
    if 'Yield_kg_per_ha' in df.columns:
        target_col = 'Yield_kg_per_ha'
        print(f"Target Column found: {target_col}")
    elif 'Yield' in df.columns:
        target_col = 'Yield'
    elif 'Production' in df.columns and 'Area' in df.columns:
        df = df[df['Area'] > 0]
        df['Yield'] = df['Production'] / df['Area']
        target_col = 'Yield'
    else:
        print("ERROR: Could not find 'Yield_kg_per_ha', 'Yield' or 'Production' target column.")
        return

    # Basic Cleaning
    df = df.dropna()
    
    # Feature Selection
    # Based on PDF and User request: Soil + Env + Crop info
    # Columns available: 'Dist Code', 'Year', 'State Code', 'State Name', 'Dist Name', 'Crop', 'Area_ha', 'Yield_kg_per_ha', ...
    # 'Temperature_C', 'Humidity_%', 'pH', 'Rainfall_mm'
    
    # We will use: State Name, Dist Name, Year, Crop, Area_ha, Temperature_C, Humidity_%, pH, Rainfall_mm
    # We will renaming them to standard keys for easier app usage
    
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
    
    # Check if these columns exist
    selected_features = []
    for csv_col, std_col in column_mapping.items():
        if csv_col in df.columns:
            df[std_col] = df[csv_col] # Rename/Copy
            selected_features.append(std_col)
        else:
            print(f"Warning: Feature '{csv_col}' not found in CSV. Ignoring.")
    
    if not selected_features:
        print("Error: No valid features found.")
        return

    X = df[selected_features]
    y = df[target_col]
    
    # Encoding
    categorical_cols = X.select_dtypes(include=['object']).columns
    print(f"Encoding categorical columns: {categorical_cols.tolist()}")
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Convert to string to ensure consistency
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        
    # Save Encoders
    joblib.dump(label_encoders, 'models/yield_label_encoders.pkl')
    
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model
    model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Yield Model R2 Score: {r2:.4f}")
    print(f"Yield Model RMSE: {rmse:.4f}")
    
    # Save Model
    joblib.dump(model, 'models/yield_model.pkl')
    # Save feature names
    joblib.dump(selected_features, 'models/yield_features.pkl')
    print("Yield Model Saved Successfully.")

def train_recommendation_model():
    print("\n--- Training Crop Recommendation Model ---")
    try:
        df = pd.read_csv('Datasets/Crop_recommendation.csv')
    except Exception as e:
        print(f"Failed to read Recommendation CSV: {e}")
        return

    print(f"Columns found: {df.columns.tolist()}")
    df.columns = df.columns.str.strip()
    
    # Target
    target_col = 'label'
    if target_col not in df.columns:
        # Try to find object column
        obj_cols = df.select_dtypes(include=['object']).columns
        if len(obj_cols) == 1:
            target_col = obj_cols[0]
            print(f"Assumed target column: {target_col}")
        else:
            print("ERROR: Could not identify target column 'label'.")
            return

    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Encode Target? 
    # RandomForestClassifier handles string labels? Yes, sklearn usually handles it, but better to encode if needed?
    # Actually sklearn classifiers *can* handle string targets, but `predict` returns strings. This is fine.
    # But usually standard practice is to rely on it.
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Recommendation Model Accuracy: {acc:.4f}")
    
    joblib.dump(model, 'models/recommend_model.pkl')
    print("Recommendation Model Saved Successfully.")

if __name__ == '__main__':
    train_yield_model()
    train_recommendation_model()
