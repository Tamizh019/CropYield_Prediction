from flask import Flask, render_template, request, send_from_directory, jsonify, session
import joblib
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv
from functools import wraps

# Load Environment Variables
load_dotenv()

# Configure Gemini AI with Pro settings
gen_ai_key = os.getenv("GOOGLE_API_KEY")
client = None
if gen_ai_key:
    client = genai.Client(api_key=gen_ai_key)
    print("✅ Gemini AI Client Configured")
else:
    print("⚠️ WARNING: GOOGLE_API_KEY not found in .env")

# Initialize Flask App
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'agrivision-secret-key-2026')

# Global variables for models
yield_model = None
yield_encoders = None
yield_scaler = None
recommend_model = None
recommend_encoders = None
recommend_le = None
recommend_scaler = None

# Statistics tracking
prediction_history = []
MAX_HISTORY = 100


# ========================================
# MODEL LOADING
# ========================================

def load_models():
    """Load all ML models at startup with error handling"""
    global yield_model, yield_encoders, yield_scaler
    global recommend_model, recommend_encoders, recommend_le, recommend_scaler
    
    try:
        # Load Yield Prediction Model
        if os.path.exists('models/yield_model.pkl'):
            yield_model = joblib.load('models/yield_model.pkl')
            yield_encoders = joblib.load('models/yield_label_encoders.pkl')
            print("✅ Yield Prediction Model Loaded")
            
            # Optional: Load scaler if exists
            if os.path.exists('models/yield_scaler.pkl'):
                yield_scaler = joblib.load('models/yield_scaler.pkl')
        else:
            print("⚠️ Yield model not found. Run training script first.")
        
        # Load Crop Recommendation Model
        if os.path.exists('models/recommend_model.pkl'):
            recommend_model = joblib.load('models/recommend_model.pkl')
            print("✅ Crop Recommendation Model Loaded")
            
            # Optional: Load encoders if exists
            if os.path.exists('models/recommend_encoders.pkl'):
                recommend_encoders = joblib.load('models/recommend_encoders.pkl')
            
            # Load Label Encoder for target
            if os.path.exists('models/recommend_label_encoder.pkl'):
                recommend_le = joblib.load('models/recommend_label_encoder.pkl')

            # Load Scaler
            if os.path.exists('models/recommend_scaler.pkl'):
                recommend_scaler = joblib.load('models/recommend_scaler.pkl')
                print("✅ Recommendation Scaler Loaded")
        else:
            print("⚠️ Recommendation model not found.")
            
    except Exception as e:
        print(f"❌ Model Loading Error: {e}")

load_models()


# ========================================
# AI FUNCTIONS
# ========================================

def get_ai_insight(data, predicted_yield):
    """
    Generate detailed agronomic insights using Gemini AI
    """
    if not client:
        return None
    
    try:
        prompt = f"""You are an expert agronomist. Analyze this agricultural prediction:

📊 FARM DATA:
Crop: {data['Crop']}
Location: {data['District_Name']}, {data['State_Name']}
Area: {data['Area']} hectares
Predicted Yield: {predicted_yield} tonnes

🌡️ ENVIRONMENTAL CONDITIONS:
Temperature: {data['Temperature']}°C
Humidity: {data['Humidity']}%
Rainfall: {data['Rainfall']}mm
Soil pH: {data['pH']}

Provide a structured analysis:

1. YIELD ASSESSMENT
[Evaluate if this yield is good/average/poor for these conditions. Compare with typical yields.]

2. OPTIMIZATION STRATEGIES
• [First specific, actionable recommendation]
• [Second specific, actionable recommendation]

3. RISK FACTORS
[Identify the main risk (weather/pH/climate) and explain its impact]

4. SEASONAL ADVICE
[Brief tip on best practices for current conditions]

Keep each section concise and farmer-friendly."""
        
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=2000,
                top_p=0.9,
                top_k=40
            )
        )
        return response.text
    
    except Exception as e:
        print(f"AI Insight Error: {e}")
        return "AI analysis temporarily unavailable. Please try again later."


def get_bulk_ai_summary(stats):
    """
    Generate comprehensive bulk dataset analysis
    """
    if not client:
        return None
    
    try:
        prompt = f"""Analyze this agricultural dataset summary:

📊 DATASET OVERVIEW:
Total Records: {stats['total_rows']}
Total Predicted Yield: {stats['total_yield']} tonnes
Average Yield: {stats['avg_yield']} tonnes/hectare
Top Performing State: {stats['top_state']}
Best Yielding Crop: {stats['top_crop']}

Provide expert insights:

1. OVERALL ASSESSMENT
[Brief evaluation of the dataset's agricultural potential]

2. REGIONAL ANALYSIS
[Key insights about {stats['top_state']}'s performance]

3. CROP RECOMMENDATIONS
[Strategic advice based on {stats['top_crop']}'s dominance]

4. ACTIONABLE INSIGHTS
[2-3 specific recommendations for farmers or policymakers]

Keep insights data-driven and practical."""
        
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=2000,
                top_p=0.9,
                top_k=40
            )
        )
        return response.text
    
    except Exception as e:
        print(f"Bulk AI Error: {e}")
        return "Bulk analysis temporarily unavailable."


def get_crop_recommendation_insight(recommended_crop, input_data):
    """
    Get AI insights for crop recommendation
    """
    if not client:
        return None
    
    try:
        prompt = f"""As an agricultural expert, provide insights for this crop recommendation:

🌾 RECOMMENDED CROP: {recommended_crop}

📊 SOIL & CLIMATE DATA:
Nitrogen (N): {input_data['N']} ppm
Phosphorus (P): {input_data['P']} ppm
Potassium (K): {input_data['K']} ppm
Temperature: {input_data['Temperature']}°C
Humidity: {input_data['Humidity']}%
pH: {input_data['pH']}
Rainfall: {input_data['Rainfall']}mm

Provide a structured, friendly, and professional analysis. Use emojis to make it engaging.

1. 🌟 WHY THIS CROP?
[Explain simply why {recommended_crop} matches the soil/conditions. Keep it under 2 sentences.]

2. 🚜 CULTIVATION TIPS
• [Tip 1]
• [Tip 2]

3. 💰 PROFITABILITY
[Brief mention of market potential or yield expectation]

Tone: Helpful expert. Keep it concise (max 100 words)."""
        
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=2000,
                top_p=0.9,
                top_k=40
            )
        )
        return response.text
    
    except Exception as e:
        print(f"Recommendation AI Error: {e}")
        return None


# ========================================
# UTILITY FUNCTIONS
# ========================================

def validate_input(data, input_type='yield'):
    """
    Validate user input data
    """
    try:
        if input_type == 'yield':
            required_fields = ['State_Name', 'District_Name', 'Crop', 'Area', 
                             'Temperature', 'Humidity', 'pH', 'Rainfall']
            
            for field in required_fields:
                if field not in data or data[field] in [None, '', 'None']:
                    return False, f"Missing or invalid field: {field}"
            
            # Range validations
            if not (0 < float(data['Area']) < 100000):
                return False, "Area must be between 0 and 100,000 hectares"
            
            if not (-10 < float(data['Temperature']) < 60):
                return False, "Temperature must be between -10°C and 60°C"
            
            if not (0 <= float(data['Humidity']) <= 100):
                return False, "Humidity must be between 0% and 100%"
            
            if not (0 <= float(data['pH']) <= 14):
                return False, "pH must be between 0 and 14"
            
            if not (0 <= float(data['Rainfall']) < 5000):
                return False, "Rainfall must be between 0 and 5000mm"
        
        return True, "Valid"
    
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def save_prediction_history(data, prediction, prediction_type='yield'):
    """
    Save prediction to history for analytics
    """
    global prediction_history
    
    try:
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': prediction_type,
            'input': data,
            'prediction': prediction
        }
        
        prediction_history.append(history_entry)
        
        # Keep only last 100 predictions
        if len(prediction_history) > MAX_HISTORY:
            prediction_history = prediction_history[-MAX_HISTORY:]
        
        # Optional: Save to file
        history_file = 'data/prediction_history.json'
        os.makedirs('data', exist_ok=True)
        
        with open(history_file, 'w') as f:
            json.dump(prediction_history, f, indent=2)
    
    except Exception as e:
        print(f"History save error: {e}")


def get_model_info():
    """
    Get information about loaded models
    """
    info = {
        'yield_model': yield_model is not None,
        'recommend_model': recommend_model is not None,
        'ai_model': client is not None,
        'prediction_count': len(prediction_history)
    }
    return info


# ========================================
# ROUTES
# ========================================

@app.route('/')
def home():
    """Home/Dashboard page"""
    model_info = get_model_info()
    return render_template('index.html', model_info=model_info)


@app.route('/predict_yield', methods=['GET', 'POST'])
def predict_yield():
    """Yield prediction route with AI insights"""
    prediction = None
    ai_insight = None
    error = None
    
    if request.method == 'POST':
        try:
            if not yield_model:
                error = "⚠️ Yield prediction model not loaded. Please train the model first."
                return render_template('predict_yield.html', error=error)
            
            # Get form data
            data = {
                'State_Name': request.form.get('State_Name'),
                'District_Name': request.form.get('District_Name'),
                'Crop_Year': float(request.form.get('Crop_Year', 2024)),
                'Crop': request.form.get('Crop'),
                'Area': float(request.form.get('Area')),
                'Temperature': float(request.form.get('Temperature')),
                'Humidity': float(request.form.get('Humidity')),
                'pH': float(request.form.get('pH')),
                'Rainfall': float(request.form.get('Rainfall'))
            }
            
            # Validate input
            is_valid, validation_msg = validate_input(data, 'yield')
            if not is_valid:
                error = f"❌ {validation_msg}"
                return render_template('predict_yield.html', error=error)
            
            # Load feature names
            if not os.path.exists('models/yield_features.pkl'):
                error = "❌ Model features file missing"
                return render_template('predict_yield.html', error=error)
            
            feature_names = joblib.load('models/yield_features.pkl')
            input_df = pd.DataFrame([data])
            
            # Encode categorical features
            for col, le in yield_encoders.items():
                if col in input_df.columns:
                    val = str(input_df.iloc[0][col])
                    if val in le.classes_:
                        input_df[col] = le.transform([val])[0]
                    else:
                        # Handle unknown categories
                        print(f"Warning: Unknown category '{val}' for {col}")
                        input_df[col] = 0
            
            # Ensure all required features exist
            for f in feature_names:
                if f not in input_df.columns:
                    input_df[f] = 0
            
            # Predict
            X_pred = input_df[feature_names]
            pred_val = yield_model.predict(X_pred)[0]
            prediction = f"{pred_val:.2f}"
            
            # Generate AI insights
            ai_insight = get_ai_insight(data, prediction)
            
            # Save to history
            save_prediction_history(data, prediction, 'yield')
            
        except ValueError as ve:
            error = f"❌ Invalid input format: {str(ve)}"
            print(f"ValueError: {ve}")
        except Exception as e:
            error = f"❌ Prediction error: {str(e)}"
            print(f"Prediction Error: {e}")
    
    return render_template('predict_yield.html', 
                         prediction=prediction, 
                         ai_insight=ai_insight,
                         error=error)


@app.route('/predict_yield_bulk', methods=['POST'])
def predict_yield_bulk():
    """Bulk yield prediction with comprehensive analytics"""
    
    if 'file' not in request.files or request.files['file'].filename == '':
        return "❌ No file uploaded", 400
    
    file = request.files['file']
    
    try:
        # Read CSV
        df = pd.read_csv(file)
        original_df = df.copy()
        
        print(f"📊 Processing {len(df)} records...")
        
        # Column mapping
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
        
        # Encode categorical features
        for col, le in yield_encoders.items():
            if col in df.columns:
                def safe_encode(x):
                    x_str = str(x)
                    return le.transform([x_str])[0] if x_str in le.classes_ else 0
                
                df[col] = df[col].apply(safe_encode)
        
        # Load features and predict
        feature_names = joblib.load('models/yield_features.pkl')
        for f in feature_names:
            if f not in df.columns:
                df[f] = 0
        
        X = df[feature_names]
        predictions = yield_model.predict(X)
        
        # Add predictions to original dataframe
        original_df['Predicted_Yield_tonnes'] = predictions.round(2)
        
        # Save output CSV
        output_path = os.path.join('static', 'predicted_yield.csv')
        os.makedirs('static', exist_ok=True)
        original_df.to_csv(output_path, index=False)
        
        # ========================================
        # ANALYTICS CALCULATIONS
        # ========================================
        
        total_yield = round(original_df['Predicted_Yield_tonnes'].sum(), 2)
        avg_yield = round(original_df['Predicted_Yield_tonnes'].mean(), 2)
        max_yield = round(original_df['Predicted_Yield_tonnes'].max(), 2)
        min_yield = round(original_df['Predicted_Yield_tonnes'].min(), 2)
        total_rows = len(original_df)
        
        # State Analysis
        state_col = next((col for col in original_df.columns 
                         if 'state' in col.lower()), None)
        state_labels, state_yields, top_state = [], [], "N/A"
        
        if state_col:
            state_grp = original_df.groupby(state_col)['Predicted_Yield_tonnes'].mean()
            state_grp = state_grp.sort_values(ascending=False).head(5)
            top_state = state_grp.index[0]
            state_labels = state_grp.index.tolist()
            state_yields = [round(y, 2) for y in state_grp.values]
        
        # Crop Analysis
        crop_col = next((col for col in original_df.columns 
                        if 'crop' in col.lower() and 'year' not in col.lower()), None)
        crop_labels, crop_counts, top_crop = [], [], "N/A"
        
        if crop_col:
            top_crop = original_df.groupby(crop_col)['Predicted_Yield_tonnes'].mean().idxmax()
            crop_dist = original_df[crop_col].value_counts().head(5)
            crop_labels = crop_dist.index.tolist()
            crop_counts = crop_dist.values.tolist()
        
        # Preview data (first 25 rows)
        preview_cols = original_df.columns.tolist()
        preview_data = original_df.head(25).values.tolist()
        
        # Generate AI Summary
        bulk_ai_insight = get_bulk_ai_summary({
            'total_rows': total_rows,
            'total_yield': total_yield,
            'avg_yield': avg_yield,
            'top_state': top_state,
            'top_crop': top_crop
        })
        
        print(f"✅ Bulk prediction complete: {total_rows} records processed")
        
        return render_template(
            'bulk_result.html',
            total_rows=total_rows,
            total_yield=f"{total_yield:,.2f}",
            avg_yield=avg_yield,
            max_yield=max_yield,
            min_yield=min_yield,
            top_state=top_state,
            top_crop=top_crop,
            state_labels=state_labels,
            state_yields=state_yields,
            crop_labels=crop_labels,
            crop_counts=crop_counts,
            columns=preview_cols,
            preview_data=preview_data,
            ai_insight=bulk_ai_insight
        )
        
    except pd.errors.EmptyDataError:
        return "❌ Error: CSV file is empty", 400
    except KeyError as ke:
        return f"❌ Error: Missing required column: {str(ke)}", 400
    except Exception as e:
        print(f"Bulk processing error: {e}")
        return f"❌ Error processing file: {str(e)}", 500


@app.route('/recommend_crop', methods=['GET', 'POST'])
def recommend_crop():
    """Crop recommendation with AI insights"""
    recommendation = None
    ai_insight = None
    error = None
    
    if request.method == 'POST':
        try:
            if not recommend_model:
                error = "⚠️ Crop recommendation model not loaded"
                return render_template('recommend.html', error=error)
            
            # Get input features
            input_data = {
                'N': float(request.form.get('N')),
                'P': float(request.form.get('P')),
                'K': float(request.form.get('K')),
                'Temperature': float(request.form.get('temperature')),
                'Humidity': float(request.form.get('humidity')),
                'pH': float(request.form.get('ph')),
                'Rainfall': float(request.form.get('rainfall'))
            }
            
            features_dict = {
                'N': input_data['N'],
                'P': input_data['P'],
                'K': input_data['K'],
                'temperature': input_data['Temperature'],
                'humidity': input_data['Humidity'],
                'ph': input_data['pH'],
                'rainfall': input_data['Rainfall']
            }
            
            # Create DataFrame for feature engineering
            X = pd.DataFrame([features_dict])
            
            # Feature Engineering (Must match training logic)
            X['NPK_Sum'] = X['N'] + X['P'] + X['K']
            X['NPK_Ratio'] = X['N'] / (X['P'] + X['K'] + 1)
            X['NK_Ratio'] = X['N'] / (X['K'] + 1)
            X['PK_Ratio'] = X['P'] / (X['K'] + 1)
            
            X['temp_humidity'] = X['temperature'] * X['humidity']
            X['rainfall_ph'] = X['rainfall'] * X['ph']
            
            # Scale features if scaler is loaded
            if recommend_scaler:
                features_scaled = recommend_scaler.transform(X)
                pred_idx = recommend_model.predict(features_scaled)[0]
            else:
                # Fallback if scaler missing (might fail if model expects scaled data)
                pred_idx = recommend_model.predict(X)[0]
            
            # Decode prediction
            if recommend_le:
                recommendation = recommend_le.inverse_transform([pred_idx])[0]
            else:
                recommendation = str(pred_idx)
            
            recommendation = recommendation.title()
            
            # Generate AI insights
            ai_insight = get_crop_recommendation_insight(recommendation, input_data)
            
            # Save to history
            save_prediction_history(input_data, recommendation, 'crop')
            
        except ValueError as ve:
            error = f"❌ Invalid input: {str(ve)}"
        except Exception as e:
            error = f"❌ Error: {str(e)}"
            print(f"Recommendation Error: {e}")
    
    return render_template('recommend.html', 
                         recommendation=recommendation,
                         ai_insight=ai_insight,
                         error=error)


@app.route('/download/<path:filename>')
def download_file(filename):
    """Download generated CSV files"""
    try:
        return send_from_directory('static', filename, as_attachment=True)
    except FileNotFoundError:
        return "File not found", 404


@app.route('/api/model_status')
def model_status():
    """API endpoint to check model status"""
    return jsonify(get_model_info())


@app.route('/api/history')
def get_history():
    """API endpoint to get prediction history"""
    return jsonify({
        'total_predictions': len(prediction_history),
        'recent': prediction_history[-10:] if prediction_history else []
    })


@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')


@app.route('/dashboard')
def dashboard():
    """Redirect to home"""
    return render_template('index.html', model_info=get_model_info())


@app.route('/documentation')
def documentation():
    """User Documentation Page"""
    return render_template('documentation.html')


@app.route('/sample_data')
def sample_data():
    """Sample Datasets Page"""
    return render_template('sample_data.html')


@app.route('/download_dataset/<path:filename>')
def download_dataset(filename):
    """Download files from Datasets folder"""
    try:
        return send_from_directory('Datasets', filename, as_attachment=True)
    except FileNotFoundError:
        return "File not found", 404


# ========================================
# ERROR HANDLERS
# ========================================

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500


# ========================================
# MAIN
# ========================================

if __name__ == '__main__':
    # Ensure required directories exist
    os.makedirs('static', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load models
    load_models()
    
    # Print startup info
    print("\n" + "="*50)
    print("🌾 AgriVision - Agricultural Intelligence Platform")
    print("="*50)
    print(f"✅ Yield Model: {'Loaded' if yield_model else 'Not Found'}")
    print(f"✅ Recommendation Model: {'Loaded' if recommend_model else 'Not Found'}")
    print(f"✅ AI Model: {'Configured' if client else 'Not Configured'}")
    print(f"📊 Prediction History: {len(prediction_history)} records")
    print("="*50 + "\n")
    
    # Run Flask app
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )
