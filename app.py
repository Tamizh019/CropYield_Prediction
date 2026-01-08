from flask import Flask, render_template, request, jsonify, send_from_directory
import flask
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Global variables for models
yield_model = None
yield_encoders = None
recommend_model = None

def load_models():
    global yield_model, yield_encoders, recommend_model
    try:
        if os.path.exists('models/yield_model.pkl'):
            yield_model = joblib.load('models/yield_model.pkl')
            yield_encoders = joblib.load('models/yield_label_encoders.pkl')
            print("Yield Model Loaded.")
        
        if os.path.exists('models/recommend_model.pkl'):
            recommend_model = joblib.load('models/recommend_model.pkl')
            print("Recommendation Model Loaded.")
    except Exception as e:
        print(f"Error loading models: {e}")

load_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_yield', methods=['GET', 'POST'])
def predict_yield():
    prediction = None
    if request.method == 'POST':
        try:
            if not yield_model:
                return render_template('predict_yield.html', prediction="Error: Model not loaded. Please train models first.")
            
            # Get form data
            data = {
                'State_Name': request.form.get('State_Name'),
                'District_Name': request.form.get('District_Name'),
                'Crop_Year': float(request.form.get('Crop_Year')),
                'Crop': request.form.get('Crop'),
                'Area': float(request.form.get('Area')),
                'Temperature': float(request.form.get('Temperature')),
                'Humidity': float(request.form.get('Humidity')),
                'pH': float(request.form.get('pH')),
                'Rainfall': float(request.form.get('Rainfall'))
            }
            
            # Create DataFrame
            df = pd.DataFrame([data])
            
            # Encode Categorical Features
            # We must use the same encoders as training
            # Handle unknown labels safely
            for col, le in yield_encoders.items():
                if col in df.columns:
                    # Check if value exists in encoder
                    val = df.iloc[0][col]
                    if val in le.classes_:
                        df[col] = le.transform([val])
                    else:
                        # Fallback or Error
                        # For now, pick the first class or 0 to avoid crash, but warn
                        # Better: Use 'handle_unknown' if using OrdinalEncoder, but LabelEncoder doesn't support it.
                        print(f"Warning: Unknown category '{val}' for column '{col}'. Using default.")
                        df[col] = 0 # transformation usually maps to integers. 0 is valid index.
            
            # Reorder columns to match training
            # Load feature names if verified
            if os.path.exists('models/yield_features.pkl'):
                feature_names = joblib.load('models/yield_features.pkl')
                # Ensure all features exist
                for f in feature_names:
                    if f not in df.columns:
                        df[f] = 0 # Missing feature
                df = df[feature_names]
            
            # Predict
            pred = yield_model.predict(df)[0]
            prediction = f"{pred:.2f}"
            
        except Exception as e:
            print(f"Prediction Error: {e}")
            prediction = f"Error: {e}"

    return render_template('predict_yield.html', prediction=prediction)

@app.route('/predict_yield_bulk', methods=['POST'])
def predict_yield_bulk():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    if file:
        try:
            df = pd.read_csv(file)
            original_df = df.copy() # Keep original to append results
            
            # Column Mapping (Same as training)
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
            
            # Rename columns
            for csv_col, std_col in column_mapping.items():
                if csv_col in df.columns:
                    df[std_col] = df[csv_col]

            # Encode Categorical Features
            for col, le in yield_encoders.items():
                if col in df.columns:
                    def safe_transform(x):
                        x = str(x)
                        if x in le.classes_:
                            return le.transform([x])[0]
                        else:
                            return 0 # Default for unknown
                    
                    df[col] = df[col].apply(safe_transform)
            
            # Feature Selection & Ordering
            if os.path.exists('models/yield_features.pkl'):
                feature_names = joblib.load('models/yield_features.pkl')
                for f in feature_names:
                    if f not in df.columns:
                        df[f] = 0
                X = df[feature_names]
            else:
                return "Model features not found. Please train model first.", 500
            
            # Predict
            predictions = yield_model.predict(X)
            
            # Append to original
            original_df['Predicted_Yield_tonnes'] = predictions
            original_df['Predicted_Yield_tonnes'] = original_df['Predicted_Yield_tonnes'].round(2)
            
            # Save CSV for download
            # Using a static filename for simplicity or unique ID in real app
            output_path = os.path.join('static', 'predicted_yield.csv')
            original_df.to_csv(output_path, index=False)
            
            # --- Analytics Logic ---
            
            # 1. Total & Avg
            total_yield = round(original_df['Predicted_Yield_tonnes'].sum(), 2)
            avg_yield = round(original_df['Predicted_Yield_tonnes'].mean(), 2)
            total_rows = len(original_df)
            
            # 2. Top State (Group By)
            # Check if State exists in original
            state_col = None
            for col in original_df.columns:
                if 'state' in col.lower():
                    state_col = col
                    break
            
            top_state = "N/A"
            state_labels = []
            state_yields = []
            
            if state_col:
                state_grp = original_df.groupby(state_col)['Predicted_Yield_tonnes'].mean().sort_values(ascending=False).head(5)
                top_state = state_grp.index[0]
                state_labels = state_grp.index.tolist()
                state_yields = state_grp.values.tolist()
                # Round yields
                state_yields = [round(y, 2) for y in state_yields]
            
            # 3. Top Crop & Distribution
            crop_col = None
            for col in original_df.columns:
                if 'crop' in col.lower() and 'year' not in col.lower(): # Avoid Crop Year
                    crop_col = col
                    break
            
            top_crop = "N/A"
            crop_labels = []
            crop_counts = []
            
            if crop_col:
                # Top yielding crop (avg)
                top_crop = original_df.groupby(crop_col)['Predicted_Yield_tonnes'].mean().idxmax()
                
                # Distribution (Count)
                crop_dist = original_df[crop_col].value_counts().head(5)
                crop_labels = crop_dist.index.tolist()
                crop_counts = crop_dist.values.tolist()

            # 4. Preview Data (Top 50)
            preview_cols = original_df.columns.tolist()
            preview_data = original_df.head(50).values.tolist()
            
            return render_template(
                'bulk_result.html',
                total_rows=total_rows,
                total_yield=f"{total_yield:,.2f}",
                avg_yield=avg_yield,
                top_state=top_state,
                top_crop=top_crop,
                state_labels=state_labels,
                state_yields=state_yields,
                crop_labels=crop_labels,
                crop_counts=crop_counts,
                columns=preview_cols,
                preview_data=preview_data
            )
            
        except Exception as e:
            return f"Error processing file: {e}", 500

@app.route('/download/<path:filename>')
def download_file(filename):
    return flask.send_from_directory('static', filename, as_attachment=True)

@app.route('/recommend_crop', methods=['GET', 'POST'])
def recommend_crop():
    recommendation = None
    if request.method == 'POST':
        try:
            if not recommend_model:
                return render_template('recommend.html', recommendation="Error: Model not loaded.")
            
            # Inputs
            features = [
                float(request.form.get('N')),
                float(request.form.get('P')),
                float(request.form.get('K')),
                float(request.form.get('temperature')),
                float(request.form.get('humidity')),
                float(request.form.get('ph')),
                float(request.form.get('rainfall'))
            ]
            
            pred = recommend_model.predict([features])[0]
            recommendation = str(pred)
            
        except Exception as e:
            print(f"Recommendation Error: {e}")
            recommendation = f"Error: {e}"

    return render_template('recommend.html', recommendation=recommendation)

@app.route('/dashboard')
def dashboard():
    return render_template('index.html') # Placeholder for dashboard content or just redirect to home

if __name__ == '__main__':
    # Load models again ensures they are fresh if script restarts
    load_models()
    app.run(debug=True, port=5000)
