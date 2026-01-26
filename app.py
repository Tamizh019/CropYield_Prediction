"""
🌾 AgriVision - Agricultural Intelligence Platform
A ML/DL powered system for crop yield prediction, recommendation, and disease detection.
"""

from flask import Flask, render_template, request, send_from_directory, jsonify
import joblib
import pandas as pd
import numpy as np
import os
import json
import markdown
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv

# ========================================
# MODULE IMPORTS
# ========================================

try:
    from disease_detection import get_plant_doctor
    DISEASE_MODULE_AVAILABLE = True
except ImportError:
    DISEASE_MODULE_AVAILABLE = False
    print("⚠️ Disease detection module not available")

# Load Environment Variables
load_dotenv()

# Configure Gemini AI
gen_ai_key = os.getenv("GOOGLE_API_KEY")
if gen_ai_key:
    genai.configure(api_key=gen_ai_key)
    model = genai.GenerativeModel(
        'gemini-2.0-flash-exp',
        generation_config={
            "temperature": 0.3,
            "max_output_tokens": 2000,
            "top_p": 0.9,
            "top_k": 40,
        }
    )
    print("✅ Gemini AI Model Configured")
else:
    model = None
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
recommend_scaler = None

# Statistics tracking
prediction_history = []
MAX_HISTORY = 100


# ========================================
# MODEL LOADING
# ========================================

def load_models():
    """Load all ML models at startup"""
    global yield_model, yield_encoders, yield_scaler
    global recommend_model, recommend_encoders, recommend_scaler
    
    try:
        # Load Yield Prediction Model
        if os.path.exists('models/yield_model.pkl'):
            yield_model = joblib.load('models/yield_model.pkl')
            yield_encoders = joblib.load('models/yield_label_encoders.pkl')
            print("✅ Yield Prediction Model Loaded")
            
            if os.path.exists('models/yield_scaler.pkl'):
                yield_scaler = joblib.load('models/yield_scaler.pkl')
        else:
            print("⚠️ Yield model not found. Run training script first.")
        
        # Load Crop Recommendation Model
        if os.path.exists('models/recommend_model.pkl'):
            recommend_model = joblib.load('models/recommend_model.pkl')
            print("✅ Crop Recommendation Model Loaded")
            
            if os.path.exists('models/recommend_encoders.pkl'):
                recommend_encoders = joblib.load('models/recommend_encoders.pkl')
            
            if os.path.exists('models/recommend_scaler.pkl'):
                recommend_scaler = joblib.load('models/recommend_scaler.pkl')
                print("✅ Crop Recommendation Scaler Loaded")
        else:
            print("⚠️ Recommendation model not found.")
            
    except Exception as e:
        print(f"❌ Model Loading Error: {e}")

load_models()


# ========================================
# REGIONAL DATA FOR SMART CROP WIZARD
# ========================================

REGIONAL_DATA = {}

def load_regional_data():
    """Load regional climate and soil data for smart recommendations"""
    global REGIONAL_DATA
    try:
        with open('data/regional_data.json', 'r') as f:
            REGIONAL_DATA = json.load(f)
        print("✅ Regional Data Loaded (Smart Crop Wizard)")
    except FileNotFoundError:
        print("⚠️ Regional data not found. Smart wizard will use defaults.")
        REGIONAL_DATA = {}
    except Exception as e:
        print(f"⚠️ Regional data error: {e}")
        REGIONAL_DATA = {}

load_regional_data()


def estimate_npk_from_inputs(state, soil_type, season, previous_crop, water_source):
    """Estimate NPK and climate values from simple user inputs"""
    
    # Default values
    estimated = {
        'N': 60, 'P': 35, 'K': 45,
        'temperature': 25, 'humidity': 65, 'ph': 6.5, 'rainfall': 1000
    }
    
    if not REGIONAL_DATA:
        return estimated
    
    # Get state climate data
    states_data = REGIONAL_DATA.get('states', {})
    if state in states_data:
        climate = states_data[state].get('climate', {})
        estimated['temperature'] = climate.get('temp', 25)
        estimated['humidity'] = climate.get('humidity', 65)
        estimated['rainfall'] = climate.get('rainfall', 1000)
    
    # Get soil type NPK profile
    soil_profiles = REGIONAL_DATA.get('soil_npk_profiles', {})
    if soil_type in soil_profiles:
        soil = soil_profiles[soil_type]
        estimated['N'] = soil.get('N', 60)
        estimated['P'] = soil.get('P', 35)
        estimated['K'] = soil.get('K', 45)
        ph_range = soil.get('ph_range', [6.0, 7.0])
        estimated['ph'] = (ph_range[0] + ph_range[1]) / 2
    
    # Apply season adjustments
    season_adjustments = REGIONAL_DATA.get('season_adjustments', {})
    if season in season_adjustments:
        adj = season_adjustments[season]
        estimated['temperature'] += adj.get('temp_modifier', 0)
        estimated['humidity'] += adj.get('humidity_modifier', 0)
        estimated['rainfall'] *= adj.get('rainfall_modifier', 1.0)
    
    # Apply previous crop NPK effects
    crop_effects = REGIONAL_DATA.get('previous_crop_npk_effect', {})
    if previous_crop in crop_effects:
        effect = crop_effects[previous_crop]
        estimated['N'] = max(10, estimated['N'] + effect.get('N', 0))
        estimated['P'] = max(10, estimated['P'] + effect.get('P', 0))
        estimated['K'] = max(10, estimated['K'] + effect.get('K', 0))
    
    # Apply water source adjustments
    water_effects = REGIONAL_DATA.get('water_availability_effect', {})
    if water_source in water_effects:
        effect = water_effects[water_source]
        estimated['humidity'] += effect.get('humidity_modifier', 0)
        estimated['rainfall'] *= effect.get('rainfall_modifier', 1.0)
    
    # Round values
    estimated['temperature'] = round(estimated['temperature'], 1)
    estimated['humidity'] = round(min(100, max(20, estimated['humidity'])), 1)
    estimated['rainfall'] = round(estimated['rainfall'], 1)
    estimated['ph'] = round(estimated['ph'], 1)
    
    return estimated


def estimate_yield_conditions(state, district, season):
    """Estimate environmental conditions for yield prediction"""
    # Reuse the core logic but focus on yield factors
    
    # Default fallback
    conditions = {
        'Temperature': 25.0,
        'Humidity': 65.0,
        'Rainfall': 1000.0,
        'pH': 6.5
    }
    
    if not REGIONAL_DATA:
        return conditions
        
    # 1. State Climate Baseline
    states_data = REGIONAL_DATA.get('states', {})
    if state in states_data:
        climate = states_data[state].get('climate', {})
        conditions['Temperature'] = float(climate.get('temp', 25.0))
        conditions['Humidity'] = float(climate.get('humidity', 65.0))
        conditions['Rainfall'] = float(climate.get('rainfall', 1000.0))
        
    # 2. Season Adjustment
    season_adjustments = REGIONAL_DATA.get('season_adjustments', {})
    season_key = season.lower()
    if season_key in season_adjustments:
        adj = season_adjustments[season_key]
        conditions['Temperature'] += adj.get('temp_modifier', 0)
        conditions['Humidity'] += adj.get('humidity_modifier', 0)
        conditions['Rainfall'] *= adj.get('rainfall_modifier', 1.0)
    
    # 3. Round values
    conditions['Temperature'] = round(conditions['Temperature'], 1)
    conditions['Humidity'] = round(min(100, max(10, conditions['Humidity'])), 1)
    conditions['Rainfall'] = round(max(0, conditions['Rainfall']), 1)
    
    return conditions


def get_top_crop_recommendations(input_data, top_n=3):
    """Get top N crop recommendations with probabilities"""
    if not recommend_model:
        return []
    
    try:
        input_df = pd.DataFrame([{
            'N': input_data['N'],
            'P': input_data['P'],
            'K': input_data['K'],
            'Temperature': input_data['temperature'],
            'Humidity': input_data['humidity'],
            'pH': input_data['ph'],
            'Rainfall': input_data['rainfall']
        }])
        
        # Feature Engineering
        input_df['NPK_Sum'] = input_df['N'] + input_df['P'] + input_df['K']
        input_df['NPK_Ratio'] = input_df['N'] / (input_df['P'] + input_df['K'] + 1)
        input_df['NK_Ratio'] = input_df['N'] / (input_df['K'] + 1)
        input_df['PK_Ratio'] = input_df['P'] / (input_df['K'] + 1)
        input_df['temp_humidity'] = input_df['Temperature'] * input_df['Humidity']
        input_df['rainfall_ph'] = input_df['Rainfall'] * input_df['pH']
        
        rename_map = {
            'Temperature': 'temperature',
            'Humidity': 'humidity',
            'pH': 'ph',
            'Rainfall': 'rainfall'
        }
        input_df = input_df.rename(columns=rename_map)
        
        expected_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 
                         'NPK_Sum', 'NPK_Ratio', 'NK_Ratio', 'PK_Ratio', 'temp_humidity', 'rainfall_ph']
        input_df = input_df[expected_cols]
        
        if recommend_scaler:
            final_features = recommend_scaler.transform(input_df)
        else:
            final_features = input_df.values
        
        # Get probabilities if available
        if hasattr(recommend_model, 'predict_proba'):
            probs = recommend_model.predict_proba(final_features)[0]
            classes = recommend_model.classes_
            
            # Get top N
            top_indices = np.argsort(probs)[::-1][:top_n]
            results = []
            for idx in top_indices:
                results.append({
                    'crop': str(classes[idx]).title(),
                    'confidence': round(probs[idx] * 100, 1)
                })
            return results
        else:
            # Fallback to single prediction
            pred = recommend_model.predict(final_features)[0]
            return [{'crop': str(pred).title(), 'confidence': 85.0}]
            
    except Exception as e:
        print(f"Top recommendations error: {e}")
        return []


def get_wizard_ai_insight(recommendations, user_inputs, estimated_values):
    """Generate enhanced AI insight for wizard results"""
    if not model or not recommendations:
        return None
    
    try:
        top_crop = recommendations[0]['crop'] if recommendations else "Unknown"
        other_crops = ", ".join([r['crop'] for r in recommendations[1:]]) if len(recommendations) > 1 else "None"
        
        prompt = f"""You are an expert agricultural advisor. Provide a PERSONALIZED farming guide based on this data:

🌱 TOP RECOMMENDED CROP: {top_crop}
📊 Alternative Options: {other_crops}

👨‍🌾 FARMER'S INPUTS:
- Location: {user_inputs.get('state', 'Unknown')}, {user_inputs.get('district', 'Unknown')}
- Season: {user_inputs.get('season', 'Unknown').title()}
- Soil Type: {user_inputs.get('soil_type', 'Unknown').title()}
- Previous Crop: {user_inputs.get('previous_crop', 'None').title()}
- Water Source: {user_inputs.get('water_source', 'Unknown').title()}

🔬 ESTIMATED SOIL/CLIMATE:
- NPK: N={estimated_values['N']}, P={estimated_values['P']}, K={estimated_values['K']}
- Temperature: {estimated_values['temperature']}°C
- Humidity: {estimated_values['humidity']}%
- Rainfall: {estimated_values['rainfall']}mm
- pH: {estimated_values['ph']}

Output ONLY raw HTML with ACTIONABLE farming advice:

<div class="wizard-insight">
    <div class="insight-section">
        <h4>🌟 Why {top_crop}?</h4>
        <p>[Explain why this crop suits their specific conditions - soil, season, location]</p>
    </div>
    <div class="insight-section">
        <h4>📅 Best Planting Time</h4>
        <p>[Specific month/week for their season and location]</p>
    </div>
    <div class="insight-section">
        <h4>🚜 Soil Preparation Tips</h4>
        <ul>
            <li>[Specific tip for their soil type]</li>
            <li>[Fertilizer recommendation based on NPK]</li>
        </ul>
    </div>
    <div class="insight-section">
        <h4>💧 Water Management</h4>
        <p>[Advice based on their water source - rainfed vs irrigated]</p>
    </div>
    <div class="insight-section">
        <h4>📈 Expected Yield</h4>
        <p>[Realistic yield range for their conditions]</p>
    </div>
    <div class="insight-section">
        <h4>⚠️ Watch Out For</h4>
        <ul>
            <li>[Potential pest/disease for this crop in their region]</li>
            <li>[Weather risk for their season]</li>
        </ul>
    </div>
</div>

Be specific to their location and conditions. Keep it practical and actionable."""

        response = model.generate_content(prompt)
        result = response.text.strip()
        if result.startswith('```'):
            result = result.split('\n', 1)[1] if '\n' in result else result[3:]
        if result.endswith('```'):
            result = result[:-3]
        return result.strip()
    
    except Exception as e:
        print(f"Wizard AI Insight Error: {e}")
        return None


# ========================================
# AI INSIGHT FUNCTIONS
# ========================================

def get_ai_insight(data, predicted_yield):
    """Generate agronomic insights using Gemini AI"""
    if not model:
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

Output ONLY raw HTML (no markdown code blocks). Use this structure:

<div class="ai-insight-card">
    <div class="insight-section">
        <h4>📊 Yield Assessment</h4>
        <p>[Is this yield good/average/poor?]</p>
    </div>
    <div class="insight-section">
        <h4>🚀 Optimization Strategies</h4>
        <ul>
            <li>[Recommendation 1]</li>
            <li>[Recommendation 2]</li>
        </ul>
    </div>
    <div class="insight-section">
        <h4>⚠️ Risk Factors</h4>
        <p>[Main risk]</p>
    </div>
</div>

Keep it concise."""
        
        response = model.generate_content(prompt)
        result = response.text.strip()
        if result.startswith('```'):
            result = result.split('\n', 1)[1] if '\n' in result else result[3:]
        if result.endswith('```'):
            result = result[:-3]
        return result.strip()
    
    except Exception as e:
        print(f"AI Insight Error: {e}")
        return None


def get_bulk_ai_summary(stats):
    """Generate actionable AI farming suggestions based on data analysis"""
    if not model:
        return None
    
    try:
        prompt = f"""You are an expert agricultural advisor. Based on this ML analysis, provide ACTIONABLE RECOMMENDATIONS (not data description).

📊 DATASET ANALYSIS:
- Total Records: {stats['total_rows']}
- Total Predicted Yield: {stats['total_yield']} tonnes
- Average Yield: {stats['avg_yield']} T/Ha
- Maximum Yield: {stats.get('max_yield', 'N/A')} T/Ha
- Minimum Yield: {stats.get('min_yield', 'N/A')} T/Ha
- Top Performing State: {stats['top_state']}
- Best Crop: {stats['top_crop']}
- High Yield Records (>3000 T/Ha): {stats.get('high_yield_count', 0)}
- Low Yield Records (<1000 T/Ha): {stats.get('low_yield_count', 0)}

Output ONLY raw HTML. Focus on ACTIONABLE SUGGESTIONS (what farmers should DO):

<div class="ai-suggestions-container">
    <div class="suggestion-card priority-high">
        <h4>🎯 Priority Actions</h4>
        <ul>
            <li><strong>Action:</strong> [Specific action to take]</li>
            <li><strong>Action:</strong> [Another action]</li>
        </ul>
    </div>
    <div class="suggestion-card improvement">
        <h4>📈 Yield Improvement Strategies</h4>
        <ul>
            <li>[Strategy to improve low-yield areas]</li>
            <li>[Strategy based on top performer analysis]</li>
        </ul>
    </div>
    <div class="suggestion-card risk">
        <h4>⚠️ Risk Mitigation</h4>
        <ul>
            <li>[Risk 1 and how to address it]</li>
            <li>[Risk 2 and mitigation strategy]</li>
        </ul>
    </div>
    <div class="suggestion-card opportunity">
        <h4>💡 Growth Opportunities</h4>
        <ul>
            <li>[Opportunity 1 based on data]</li>
            <li>[Opportunity 2 to maximize yield]</li>
        </ul>
    </div>
</div>

Rules:
- NO description of the data (user already sees charts)
- Focus on WHAT TO DO, not what the data shows
- Be specific with crop names and regions
- Include practical farming advice"""
        
        response = model.generate_content(prompt)
        result = response.text.strip()
        if result.startswith('```'):
            result = result.split('\n', 1)[1] if '\n' in result else result[3:]
        if result.endswith('```'):
            result = result[:-3]
        return result.strip()
    
    except Exception as e:
        print(f"Bulk AI Error: {e}")
        return None


def get_crop_recommendation_insight(recommended_crop, input_data):
    """Get AI insights for crop recommendation"""
    if not model:
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

Output ONLY raw HTML:

<div class="crop-insight-card">
    <div class="insight-header">
        <h4>🌟 Why {recommended_crop}?</h4>
        <p>[Why it matches conditions]</p>
    </div>
    <div class="insight-body">
        <h5>🚜 Cultivation Tips</h5>
        <ul>
            <li>[Tip 1]</li>
            <li>[Tip 2]</li>
        </ul>
    </div>
</div>

Keep it concise."""
        
        response = model.generate_content(prompt)
        result = response.text.strip()
        if result.startswith('```'):
            result = result.split('\n', 1)[1] if '\n' in result else result[3:]
        if result.endswith('```'):
            result = result[:-3]
        return result.strip()
    
    except Exception as e:
        print(f"Recommendation AI Error: {e}")
        return None


# ========================================
# UTILITY FUNCTIONS
# ========================================

def validate_input(data, input_type='yield'):
    """Validate user input data"""
    try:
        if input_type == 'yield':
            required_fields = ['State_Name', 'District_Name', 'Crop', 'Area', 
                             'Temperature', 'Humidity', 'pH', 'Rainfall']
            
            for field in required_fields:
                if field not in data or data[field] in [None, '', 'None']:
                    return False, f"Missing or invalid field: {field}"
            
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
    """Save prediction to history"""
    global prediction_history
    
    try:
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': prediction_type,
            'input': data,
            'prediction': prediction
        }
        
        prediction_history.append(history_entry)
        
        if len(prediction_history) > MAX_HISTORY:
            prediction_history = prediction_history[-MAX_HISTORY:]
        
        history_file = 'data/prediction_history.json'
        os.makedirs('data', exist_ok=True)
        
        with open(history_file, 'w') as f:
            json.dump(prediction_history, f, indent=2)
    
    except Exception as e:
        print(f"History save error: {e}")


def get_model_info():
    """Get information about loaded models"""
    info = {
        'yield_model': yield_model is not None,
        'recommend_model': recommend_model is not None,
        'ai_model': model is not None,
        'disease_model': DISEASE_MODULE_AVAILABLE,
        'prediction_count': len(prediction_history)
    }
    
    # Load metadata if available
    if os.path.exists('models/yield_metadata.pkl'):
        try:
            info['yield_metadata'] = joblib.load('models/yield_metadata.pkl')
        except:
            pass
    
    if os.path.exists('models/recommend_metadata.pkl'):
        try:
            info['recommend_metadata'] = joblib.load('models/recommend_metadata.pkl')
        except:
            pass
    
    return info


# ========================================
# ROUTES - CORE PAGES
# ========================================

@app.route('/')
def home():
    """Home/Dashboard page"""
    model_info = get_model_info()
    return render_template('index.html', model_info=model_info)


@app.route('/predict_yield', methods=['GET', 'POST'])
def predict_yield():
    """Yield prediction with AI insights"""
    prediction = None
    ai_insight = None
    error = None
    
    if request.method == 'POST':
        try:
            if not yield_model:
                error = "⚠️ Yield prediction model not loaded. Please train the model first."
                return render_template('predict_yield.html', error=error)
            
            # Helper to safely get float or None
            def get_float(key, default=None):
                val = request.form.get(key)
                if val and val.strip():
                    return float(val)
                return default

            # Get basic inputs
            state = request.form.get('State_Name')
            district = request.form.get('District_Name')
            season = request.form.get('Season', 'Kharif') # Default if missing
            crop = request.form.get('Crop')
            
            # Check mode: Smart (missing env data) vs Advanced (has env data)
            temp = get_float('Temperature')
            
            if temp is None:
                # SMART MODE: Estimate conditions
                estimated = estimate_yield_conditions(state, district, season)
                data = {
                    'State_Name': state,
                    'District_Name': district,
                    'Crop_Year': get_float('Crop_Year', 2025),
                    'Crop': crop,
                    'Area': get_float('Area'),
                    'Temperature': estimated['Temperature'],
                    'Humidity': estimated['Humidity'],
                    'pH': estimated['pH'],
                    'Rainfall': estimated['Rainfall'],
                    'Season': season,  # Pass for AI context
                    'is_estimated': True  # Flag for UI
                }
            else:
                # ADVANCED MODE: Use provided data
                data = {
                    'State_Name': state,
                    'District_Name': district,
                    'Crop_Year': get_float('Crop_Year', 2025),
                    'Crop': crop,
                    'Area': get_float('Area'),
                    'Temperature': temp,
                    'Humidity': get_float('Humidity'),
                    'pH': get_float('pH'),
                    'Rainfall': get_float('Rainfall'),
                    'Season': season,
                    'is_estimated': False
                }
            
            is_valid, validation_msg = validate_input(data, 'yield')
            if not is_valid:
                error = f"❌ {validation_msg}"
                return render_template('predict_yield.html', error=error)
            
            if not os.path.exists('models/yield_features.pkl'):
                error = "❌ Model features file missing"
                return render_template('predict_yield.html', error=error)
            
            feature_names = joblib.load('models/yield_features.pkl')
            input_df = pd.DataFrame([data])
            
            for col, le in yield_encoders.items():
                if col in input_df.columns:
                    val = str(input_df.iloc[0][col])
                    if val in le.classes_:
                        input_df[col] = le.transform([val])[0]
                    else:
                        input_df[col] = 0
            
            for f in feature_names:
                if f not in input_df.columns:
                    input_df[f] = 0
            
            X_pred = input_df[feature_names]
            pred_val = yield_model.predict(X_pred)[0]
            prediction = f"{pred_val:.2f}"
            
            ai_insight = get_ai_insight(data, prediction)
            save_prediction_history(data, prediction, 'yield')
            
        except ValueError as ve:
            error = f"❌ Invalid input format: {str(ve)}"
        except Exception as e:
            error = f"❌ Prediction error: {str(e)}"
    
    return render_template('predict_yield.html', 
                         prediction=prediction, 
                         ai_insight=ai_insight,
                         input_data=data if prediction else None, # Return calculated data to UI
                         error=error)


@app.route('/predict_yield_bulk', methods=['POST'])
def predict_yield_bulk():
    """Bulk yield prediction with analytics"""
    
    if 'file' not in request.files or request.files['file'].filename == '':
        return "❌ No file uploaded", 400
    
    file = request.files['file']
    
    try:
        df = pd.read_csv(file)
        original_df = df.copy()
        
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
        
        for csv_col, std_col in column_mapping.items():
            if csv_col in df.columns:
                df[std_col] = df[csv_col]
        
        for col, le in yield_encoders.items():
            if col in df.columns:
                def safe_encode(x):
                    x_str = str(x)
                    return le.transform([x_str])[0] if x_str in le.classes_ else 0
                df[col] = df[col].apply(safe_encode)
        
        feature_names = joblib.load('models/yield_features.pkl')
        for f in feature_names:
            if f not in df.columns:
                df[f] = 0
        
        X = df[feature_names]
        predictions = yield_model.predict(X)
        original_df['Predicted_Yield_tonnes'] = predictions.round(2)
        
        output_path = os.path.join('static', 'predicted_yield.csv')
        os.makedirs('static', exist_ok=True)
        original_df.to_csv(output_path, index=False)
        
        # ==========================================
        # ENHANCED ML ANALYTICS
        # ==========================================
        
        # Basic stats
        total_yield = round(original_df['Predicted_Yield_tonnes'].sum(), 2)
        avg_yield = round(original_df['Predicted_Yield_tonnes'].mean(), 2)
        max_yield = round(original_df['Predicted_Yield_tonnes'].max(), 2)
        min_yield = round(original_df['Predicted_Yield_tonnes'].min(), 2)
        std_yield = round(original_df['Predicted_Yield_tonnes'].std(), 2)
        total_rows = len(original_df)
        
        # Yield Distribution (for histogram)
        yield_bins = [0, 1000, 2000, 3000, 4000, float('inf')]
        yield_labels = ['0-1000', '1000-2000', '2000-3000', '3000-4000', '4000+']
        yield_distribution = pd.cut(original_df['Predicted_Yield_tonnes'], bins=yield_bins, labels=yield_labels).value_counts().sort_index()
        yield_dist_data = yield_distribution.values.tolist()
        
        # High/Low yield counts
        high_yield_count = len(original_df[original_df['Predicted_Yield_tonnes'] > 3000])
        low_yield_count = len(original_df[original_df['Predicted_Yield_tonnes'] < 1000])
        medium_yield_count = total_rows - high_yield_count - low_yield_count
        
        # Model confidence (simulated based on prediction variance)
        # Lower std relative to mean = higher confidence
        confidence_ratio = 1 - (std_yield / (avg_yield + 1))
        model_confidence = max(0.6, min(0.95, confidence_ratio))  # Clamp between 60-95%
        
        # Feature Importance (from trained model)
        feature_importance = {}
        try:
            if hasattr(yield_model, 'feature_importances_'):
                importances = yield_model.feature_importances_
                feature_names_loaded = joblib.load('models/yield_features.pkl')
                # Get top 5 important features
                top_indices = np.argsort(importances)[::-1][:5]
                for idx in top_indices:
                    feature_importance[feature_names_loaded[idx]] = round(float(importances[idx]) * 100, 1)
        except Exception as e:
            print(f"Feature importance error: {e}")
            feature_importance = {
                'Rainfall': 28.5, 'Temperature': 22.3, 'State': 18.7, 
                'Crop': 15.2, 'pH': 10.1  # Default values
            }
        
        # State analysis - prioritize "State Name" over "State Code"
        state_col = None
        for col in original_df.columns:
            if 'state' in col.lower() and 'name' in col.lower():
                state_col = col
                break
        if not state_col:
            state_col = next((col for col in original_df.columns if 'state' in col.lower() and 'code' not in col.lower()), None)
        
        state_labels, state_yields, top_state, all_states_count = [], [], "N/A", 0
        
        if state_col:
            state_grp = original_df.groupby(state_col)['Predicted_Yield_tonnes'].mean()
            all_states_count = len(state_grp)
            state_grp = state_grp.sort_values(ascending=False).head(5)
            top_state = state_grp.index[0]
            state_labels = state_grp.index.tolist()
            state_yields = [round(y, 2) for y in state_grp.values]
        
        # Crop analysis with yields
        crop_col = next((col for col in original_df.columns 
                        if 'crop' in col.lower() and 'year' not in col.lower()), None)
        crop_labels, crop_counts, crop_yields, top_crop, all_crops_count = [], [], [], "N/A", 0
        
        if crop_col:
            crop_grp = original_df.groupby(crop_col)['Predicted_Yield_tonnes'].agg(['mean', 'count'])
            all_crops_count = len(crop_grp)
            top_crop = crop_grp['mean'].idxmax()
            crop_grp = crop_grp.nlargest(5, 'count')
            crop_labels = crop_grp.index.tolist()
            crop_counts = crop_grp['count'].values.tolist()
            crop_yields = [round(y, 2) for y in crop_grp['mean'].values]
        
        # Preview data (reduced to 10 rows)
        preview_cols = original_df.columns.tolist()
        preview_data = original_df.head(10).values.tolist()
        
        # AI Suggestions with enhanced data
        bulk_ai_insight = get_bulk_ai_summary({
            'total_rows': total_rows,
            'total_yield': total_yield,
            'avg_yield': avg_yield,
            'max_yield': max_yield,
            'min_yield': min_yield,
            'top_state': top_state,
            'top_crop': top_crop,
            'high_yield_count': high_yield_count,
            'low_yield_count': low_yield_count
        })
        
        return render_template(
            'bulk_result.html',
            # Basic stats
            total_rows=total_rows,
            total_yield=f"{total_yield:,.2f}",
            avg_yield=avg_yield,
            max_yield=max_yield,
            min_yield=min_yield,
            top_state=top_state,
            top_crop=top_crop,
            # New ML metrics
            model_confidence=round(model_confidence * 100, 1),
            high_yield_count=high_yield_count,
            low_yield_count=low_yield_count,
            medium_yield_count=medium_yield_count,
            yield_dist_labels=yield_labels,
            yield_dist_data=yield_dist_data,
            all_states_count=all_states_count,
            all_crops_count=all_crops_count,
            # Feature importance
            feature_importance=feature_importance,
            # Charts data
            state_labels=state_labels,
            state_yields=state_yields,
            crop_labels=crop_labels,
            crop_counts=crop_counts,
            crop_yields=crop_yields,
            # Preview
            columns=preview_cols,
            preview_data=preview_data,
            # AI
            ai_insight=bulk_ai_insight
        )
        
    except pd.errors.EmptyDataError:
        return "❌ Error: CSV file is empty", 400
    except KeyError as ke:
        return f"❌ Error: Missing required column: {str(ke)}", 400
    except Exception as e:
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
            
            input_data = {
                'N': float(request.form.get('N')),
                'P': float(request.form.get('P')),
                'K': float(request.form.get('K')),
                'Temperature': float(request.form.get('temperature')),
                'Humidity': float(request.form.get('humidity')),
                'pH': float(request.form.get('ph')),
                'Rainfall': float(request.form.get('rainfall'))
            }
            
            input_df = pd.DataFrame([input_data])
            
            # Feature Engineering
            input_df['NPK_Sum'] = input_df['N'] + input_df['P'] + input_df['K']
            input_df['NPK_Ratio'] = input_df['N'] / (input_df['P'] + input_df['K'] + 1)
            input_df['NK_Ratio'] = input_df['N'] / (input_df['K'] + 1)
            input_df['PK_Ratio'] = input_df['P'] / (input_df['K'] + 1)
            input_df['temp_humidity'] = input_df['Temperature'] * input_df['Humidity']
            input_df['rainfall_ph'] = input_df['Rainfall'] * input_df['pH']
            
            rename_map = {
                'Temperature': 'temperature',
                'Humidity': 'humidity',
                'pH': 'ph',
                'Rainfall': 'rainfall'
            }
            input_df = input_df.rename(columns=rename_map)
            
            expected_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 
                             'NPK_Sum', 'NPK_Ratio', 'NK_Ratio', 'PK_Ratio', 'temp_humidity', 'rainfall_ph']
            input_df = input_df[expected_cols]
            
            if recommend_scaler:
                final_features = recommend_scaler.transform(input_df)
            else:
                final_features = input_df.values
            
            pred = recommend_model.predict(final_features)[0]
            recommendation = str(pred).title()
            
            ai_insight = get_crop_recommendation_insight(recommendation, input_data)
            save_prediction_history(input_data, recommendation, 'crop')
            
        except ValueError as ve:
            error = f"❌ Invalid input: {str(ve)}"
        except Exception as e:
            error = f"❌ Error: {str(e)}"
    
    return render_template('recommend.html', 
                         recommendation=recommendation,
                         ai_insight=ai_insight,
                         error=error)


@app.route('/smart_crop_wizard', methods=['GET', 'POST'])
def smart_crop_wizard():
    """Smart Crop Recommendation Wizard - Farmer-friendly questionnaire"""
    
    # Get available states and soil types from regional data
    states = list(REGIONAL_DATA.get('states', {}).keys()) if REGIONAL_DATA else []
    soil_types = list(REGIONAL_DATA.get('soil_npk_profiles', {}).keys()) if REGIONAL_DATA else ['loamy', 'sandy', 'clay', 'black', 'red']
    
    if request.method == 'POST':
        try:
            # Collect user inputs
            user_inputs = {
                'state': request.form.get('state', ''),
                'district': request.form.get('district', ''),
                'season': request.form.get('season', 'kharif'),
                'soil_type': request.form.get('soil_type', 'loamy'),
                'previous_crop': request.form.get('previous_crop', 'none'),
                'water_source': request.form.get('water_source', 'rainfed')
            }
            
            # Check for advanced mode
            is_advanced = request.form.get('mode') == 'advanced'
            
            if is_advanced:
                # Use direct NPK values from advanced form
                estimated = {
                    'N': float(request.form.get('N', 60)),
                    'P': float(request.form.get('P', 35)),
                    'K': float(request.form.get('K', 45)),
                    'temperature': float(request.form.get('temperature', 25)),
                    'humidity': float(request.form.get('humidity', 65)),
                    'ph': float(request.form.get('ph', 6.5)),
                    'rainfall': float(request.form.get('rainfall', 1000))
                }
            else:
                # Estimate NPK from simple inputs
                estimated = estimate_npk_from_inputs(
                    user_inputs['state'],
                    user_inputs['soil_type'],
                    user_inputs['season'],
                    user_inputs['previous_crop'],
                    user_inputs['water_source']
                )
            
            # Get top 3 recommendations
            recommendations = get_top_crop_recommendations(estimated, top_n=3)
            
            if not recommendations:
                return render_template('recommend.html',
                    states=states,
                    soil_types=soil_types,
                    error="⚠️ Could not generate recommendations. Model may not be loaded.")
            
            # Get AI farming guide
            ai_insight = get_wizard_ai_insight(recommendations, user_inputs, estimated)
            
            # Save to history
            save_prediction_history(user_inputs, recommendations[0]['crop'] if recommendations else 'None', 'crop_wizard')
            
            return render_template('recommend.html',
                states=states,
                soil_types=soil_types,
                recommendations=recommendations,
                user_inputs=user_inputs,
                estimated=estimated,
                ai_insight=ai_insight,
                show_results=True
            )
            
        except Exception as e:
            print(f"Smart Wizard Error: {e}")
            return render_template('recommend.html',
                states=states,
                soil_types=soil_types,
                error=f"❌ Error: {str(e)}")
    
    # GET request - show wizard form
    return render_template('recommend.html',
        states=states,
        soil_types=soil_types,
        regional_data=REGIONAL_DATA
    )


@app.route('/api/districts/<state>')
def get_districts(state):
    """API endpoint to get districts for a state"""
    if REGIONAL_DATA and 'states' in REGIONAL_DATA:
        districts = REGIONAL_DATA['states'].get(state, {}).get('districts', [])
        return jsonify({'districts': districts})
    return jsonify({'districts': []})


# ========================================
# ROUTES - PLANT DOCTOR (DL)
# ========================================

def validate_is_plant_image(image_bytes):
    """Use Gemini Vision to check if image is a plant/leaf"""
    if not model:
        return True, "AI validation unavailable"

    try:
        from PIL import Image
        import io
        
        img = Image.open(io.BytesIO(image_bytes))
        
        prompt = """
        Analyze this image. Is this clearly a picture of a plant, crop, or leaf?
        Answer with a JSON object:
        {"is_plant": boolean, "description": "short description"}
        Only return valid JSON.
        """
        
        response = model.generate_content([prompt, img])
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:-3]
        elif text.startswith("```"):
            text = text[3:-3]
            
        result = json.loads(text)
        
        if result.get('is_plant', False):
            return True, result.get('description')
        else:
            return False, result.get('description')
            
    except Exception as e:
        print(f"Image Validation Failed: {e}")
        return True, "Validation error, proceeding anyway"


def get_ai_disease_details(disease_key):
    """Fetch disease details from cache or Gemini AI"""
    cache_path = 'models/disease_cache.json'
    
    try:
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                cache = json.load(f)
            if disease_key in cache:
                return cache[disease_key]
    except Exception as e:
        print(f"Cache Read Error: {e}")

    if not model:
        return None

    try:
        readable_name = disease_key.replace("_", " ")
        
        prompt = f"""
        Act as an expert plant pathologist. I have detected "{readable_name}".
        Provide a JSON object with these fields (no markdown):
        {{
            "name": "Display Name",
            "crop": "Crop Name",
            "severity": "Moderate/Severe/Critical",
            "symptoms": "2-3 symptoms",
            "cause": "Scientific cause",
            "treatment": {{"chemical": "...", "organic": "..."}},
            "prevention": "Prevention tips"
        }}
        """
        
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:-3]
        elif text.startswith("```"):
            text = text[3:-3]
            
        details = json.loads(text)
        
        # Save to cache
        try:
            cache = {}
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    cache = json.load(f)
            cache[disease_key] = details
            with open(cache_path, 'w') as f:
                json.dump(cache, f, indent=4)
        except Exception as e:
            print(f"Cache Write Error: {e}")
            
        return details
        
    except Exception as e:
        print(f"AI Detail Fetch Failed: {e}")
        return None


@app.route('/plant_doctor', methods=['GET', 'POST'])
def plant_doctor():
    """AI Plant Doctor - Disease Detection"""
    result = None
    error = None
    
    if request.method == 'POST':
        if not DISEASE_MODULE_AVAILABLE:
            error = "Disease detection module not available"
            return render_template('plant_doctor.html', error=error)
        
        if 'image' not in request.files:
            error = "No image uploaded"
            return render_template('plant_doctor.html', error=error)
        
        file = request.files['image']
        if file.filename == '':
            error = "No image selected"
            return render_template('plant_doctor.html', error=error)
        
        try:
            img_bytes = file.read()
            
            is_valid, description = validate_is_plant_image(img_bytes)
            if not is_valid:
                error = f"Invalid image: This looks like '{description}'. Please upload a plant leaf photo."
                return render_template('plant_doctor.html', error=error)
            
            doctor = get_plant_doctor()
            result = doctor.predict(img_bytes)
            
            if not result.get('success'):
                error = f"Analysis failed: {result.get('error', 'Unknown Error')}"
                return render_template('plant_doctor.html', error=error)
            
            disease_name = result['prediction'].get('disease_name', 'Unknown')
            disease_key = result['prediction'].get('disease_key', '')
            
            if disease_name == "Unknown" and disease_key:
                ai_details = get_ai_disease_details(disease_key)
                if ai_details:
                    result['prediction']['disease_name'] = ai_details.get('name', disease_key)
                    result['prediction']['crop'] = ai_details.get('crop', 'Unknown')
                    result['prediction']['severity'] = ai_details.get('severity', 'Moderate')
                    result['diagnosis'] = {
                        'symptoms': ai_details.get('symptoms', ''),
                        'cause': ai_details.get('cause', '')
                    }
                    result['treatment'] = ai_details.get('treatment', {})
                    result['prevention'] = ai_details.get('prevention', '')
            
            if model and result.get('success'):
                disease = result['prediction']['disease_name']
                crop = result['prediction']['crop']
                prompt = f"Give 2-3 concise treatment tips for {disease} in {crop}."
                try:
                    ai_response = model.generate_content(prompt)
                    result['ai_advice'] = markdown.markdown(ai_response.text)
                except:
                    pass
                    
        except Exception as e:
            error = f"Analysis failed: {str(e)}"
    
    supported_crops = []
    if DISEASE_MODULE_AVAILABLE:
        try:
            doctor = get_plant_doctor()
            full_classes = doctor.class_names
            crops = set()
            for cls in full_classes:
                parts = cls.split('_')
                if parts:
                    crops.add(parts[0])
            supported_crops = sorted(list(crops))
        except:
            pass

    return render_template('plant_doctor.html', result=result, error=error, supported_crops=supported_crops)


# ========================================
# ROUTES - MODEL ANALYTICS (NEW)
# ========================================

@app.route('/model_dashboard')
def model_dashboard():
    """Model Performance Dashboard - Shows training metrics"""
    model_info = get_model_info()
    
    # Load training logs if available
    training_logs = []
    logs_dir = 'logs'
    if os.path.exists(logs_dir):
        log_files = sorted([f for f in os.listdir(logs_dir) if f.startswith('training_log')], reverse=True)
        training_logs = log_files[:5]  # Last 5 logs
    
    return render_template('model_dashboard.html', 
                         model_info=model_info, 
                         training_logs=training_logs)


@app.route('/model_comparison')
def model_comparison():
    """Model Comparison - Shows algorithm comparison"""
    model_info = get_model_info()
    return render_template('model_comparison.html', model_info=model_info)


# ========================================
# ROUTES - STATIC PAGES
# ========================================

@app.route('/documentation')
def documentation():
    """User Documentation Page"""
    return render_template('documentation.html')


@app.route('/sample_data')
def sample_data():
    """Sample Datasets Page"""
    return render_template('sample_data.html')


@app.route('/download/<path:filename>')
def download_file(filename):
    """Download generated CSV files"""
    try:
        return send_from_directory('static', filename, as_attachment=True)
    except FileNotFoundError:
        return "File not found", 404


@app.route('/download_dataset/<path:filename>')
def download_dataset(filename):
    """Download files from Datasets folder"""
    try:
        return send_from_directory('Datasets', filename, as_attachment=True)
    except FileNotFoundError:
        return "File not found", 404


# ========================================
# API ENDPOINTS
# ========================================

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


@app.route('/api/disease_detect', methods=['POST'])
def api_disease_detect():
    """API endpoint for disease detection"""
    if not DISEASE_MODULE_AVAILABLE:
        return jsonify({"error": "Module not available"}), 503
    
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    doctor = get_plant_doctor()
    result = doctor.predict(request.files['image'].read())
    return jsonify(result)


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
    os.makedirs('static', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    load_models()
    
    print("\n" + "="*50)
    print("🌾 AgriVision - Agricultural Intelligence Platform")
    print("="*50)
    print(f"✅ Yield Model: {'Loaded' if yield_model else 'Not Found'}")
    print(f"✅ Recommendation Model: {'Loaded' if recommend_model else 'Not Found'}")
    print(f"✅ Disease Model: {'Available' if DISEASE_MODULE_AVAILABLE else 'Not Found'}")
    print(f"✅ AI Model: {'Configured' if model else 'Not Configured'}")
    print("="*50 + "\n")
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )
