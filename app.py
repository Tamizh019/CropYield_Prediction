"""
🌾 AgriVision - YieldMax Precision Model
Advanced Multi-Algorithm Intelligence for Crop Yield Prediction & Recommendation
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

# Load Environment Variables
load_dotenv()
gen_ai_key = os.getenv("GOOGLE_API_KEY")
if gen_ai_key:
    genai.configure(api_key=gen_ai_key)
    gemini_model = genai.GenerativeModel(
        'gemini-3.1-pro-preview',       
        generation_config={
            "temperature": 0.3,
            "max_output_tokens": 4000,
            "top_p": 0.9,
            "top_k": 40,
        }
    )
    print(f"✅ Gemini AI Model Configured: gemini-3.1-pro-preview | Key ends: ...{gen_ai_key[-6:]}")
else:
    gemini_model = None
    print("⚠️ WARNING: GOOGLE_API_KEY not found in .env — AI features disabled")


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
recommend_classes = None

# Ensemble model
ensemble_model = None

# Statistics tracking
prediction_history = []
MAX_HISTORY = 100


# ========================================
# MODEL LOADING
# ========================================

def load_models():
    """Load all ML models at startup"""
    global yield_model, yield_encoders, yield_scaler
    global recommend_model, recommend_encoders, recommend_scaler, recommend_classes
    global ensemble_model
    
    try:
        # Always load label encoders if available (needed for both ensemble and single model)
        if os.path.exists('models/yield_label_encoders.pkl'):
            yield_encoders = joblib.load('models/yield_label_encoders.pkl')
            print("✅ Label Encoders Loaded")
        else:
            yield_encoders = {}
            print("⚠️ No label encoders found")
        
        # Try loading ensemble model first (YieldMax)
        if os.path.exists('models/yieldmax_ensemble.pkl'):
            try:
                from ensemble_model import YieldMaxEnsemble
                temp_model = YieldMaxEnsemble()
                temp_model.load('models/yieldmax_ensemble.pkl')
                ensemble_model = temp_model  # Only assign if load succeeds
                print("✅ YieldMax Ensemble Model Loaded")
            except Exception as e:
                print(f"⚠️ Failed to load ensemble model: {e}")
                ensemble_model = None
        
        # Load single yield model as fallback
        if os.path.exists('models/yield_model.pkl'):
            yield_model = joblib.load('models/yield_model.pkl')
            print("✅ Yield Prediction Model Loaded (fallback)")
            
            if os.path.exists('models/yield_scaler.pkl'):
                yield_scaler = joblib.load('models/yield_scaler.pkl')
        else:
            print("⚠️ Single yield model not found (using ensemble instead).")
        
        # Load Crop Recommendation Model
        if os.path.exists('models/recommend_model.pkl'):
            recommend_model = joblib.load('models/recommend_model.pkl')
            print("✅ Crop Recommendation Model Loaded")
            
            if os.path.exists('models/recommend_metadata.pkl'):
                try:
                    meta = joblib.load('models/recommend_metadata.pkl')
                    recommend_classes = meta.get('classes', [])
                    print(f"✅ Crop Recommendation Metadata Loaded ({len(recommend_classes)} crops)")
                except Exception as me:
                    print(f"⚠️ Failed to load recommend metadata: {me}")
            
            if os.path.exists('models/recommend_encoders.pkl'):
                recommend_encoders = joblib.load('models/recommend_encoders.pkl')
            
            if os.path.exists('models/recommend_scaler.pkl'):
                recommend_scaler = joblib.load('models/recommend_scaler.pkl')
                print("✅ Crop Recommendation Scaler Loaded")
        else:
            print("⚠️ Recommendation model not found.")
            
    except Exception as e:
        print(f"❌ Model Loading Error: {e}")
        import traceback
        traceback.print_exc()

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
    
    conditions = {
        'Temperature': 25.0,
        'Humidity': 65.0,
        'Rainfall': 1000.0,
        'pH': 6.5
    }
    
    if not REGIONAL_DATA:
        return conditions
        
    # State Climate Baseline
    states_data = REGIONAL_DATA.get('states', {})
    if state in states_data:
        climate = states_data[state].get('climate', {})
        conditions['Temperature'] = float(climate.get('temp', 25.0))
        conditions['Humidity'] = float(climate.get('humidity', 65.0))
        conditions['Rainfall'] = float(climate.get('rainfall', 1000.0))
        
    # Season Adjustment
    season_adjustments = REGIONAL_DATA.get('season_adjustments', {})
    season_key = season.lower()
    if season_key in season_adjustments:
        adj = season_adjustments[season_key]
        conditions['Temperature'] += adj.get('temp_modifier', 0)
        conditions['Humidity'] += adj.get('humidity_modifier', 0)
        conditions['Rainfall'] *= adj.get('rainfall_modifier', 1.0)
    
    # Round values
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
            
            top_indices = np.argsort(probs)[::-1][:top_n]
            results = []
            for idx in top_indices:
                class_id = int(classes[idx])
                if recommend_classes and len(recommend_classes) > class_id:
                    crop_name = recommend_classes[class_id]
                else:
                    crop_name = f"Crop {class_id}"
                    
                results.append({
                    'crop': str(crop_name).title(),
                    'confidence': round(probs[idx] * 100, 1)
                })
            return results
        else:
            pred = int(recommend_model.predict(final_features)[0])
            if recommend_classes and len(recommend_classes) > pred:
                crop_name = recommend_classes[pred]
            else:
                crop_name = f"Crop {pred}"
            return [{'crop': str(crop_name).title(), 'confidence': 85.0}]
            
    except Exception as e:
        print(f"Top recommendations error: {e}")
        return []


def get_wizard_ai_insight(recommendations, user_inputs, estimated_values):
    """Generate enhanced AI insight for wizard results"""
    if not gemini_model or not recommendations:
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
        <p>[Explain why this crop suits their specific conditions]</p>
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
        <p>[Advice based on their water source]</p>
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

        response = gemini_model.generate_content(prompt)
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

def get_ai_insight(data, predicted_yield, confidence=None, prediction_range=None):
    """Generate agronomic insights using Gemini AI"""
    print(f"\n{'='*50}")
    print(f"🤖 AI INSIGHT REQUEST: {data.get('Crop')} | {data.get('State_Name')}")
    
    if not gemini_model:
        print("❌ SKIP: gemini_model is None — GOOGLE_API_KEY missing")
        print('='*50)
        return None
    
    try:
        confidence_text = f" | Confidence: {confidence}%" if confidence else ""
        range_text = f" | Range: {prediction_range['lower']:.2f}-{prediction_range['upper']:.2f} T/Ha" if prediction_range else ""
        
        prompt = f"""You are an expert agronomist. A farmer in {data.get('District_Name', 'India')}, {data.get('State_Name', 'India')} is growing {data.get('Crop', 'crops')}.

Predicted yield: {predicted_yield} T/Ha{confidence_text}{range_text}
Area: {data.get('Area', 'N/A')} ha | Temperature: {data.get('Temperature', 'N/A')}°C | Humidity: {data.get('Humidity', 'N/A')}% | Rainfall: {data.get('Rainfall', 'N/A')}mm | pH: {data.get('pH', 'N/A')}

Provide a highly structured, visually appealing agronomic analysis as raw HTML only (no markdown, no code fences).
Use this EXACT HTML structure with inline styles for beautiful rendering:

<div style="display: flex; flex-direction: column; gap: 15px;">
    <!-- Yield Assessment -->
    <div style="background: rgba(52, 152, 219, 0.1); border-left: 4px solid #3498db; padding: 15px; border-radius: 8px;">
        <h4 style="color: #3498db; margin: 0 0 10px 0; font-size: 16px;">📊 Yield Assessment</h4>
        <p style="margin: 0; color: #ddd; font-size: 14px; line-height: 1.5;">[1-2 sentences: is {predicted_yield} T/Ha good/average/poor for {data.get('Crop')} in this region?]</p>
    </div>

    <!-- Optimization -->
    <div style="background: rgba(46, 204, 113, 0.1); border-left: 4px solid #2ecc71; padding: 15px; border-radius: 8px;">
        <h4 style="color: #2ecc71; margin: 0 0 10px 0; font-size: 16px;">🚀 Top 3 Optimization Tips</h4>
        <ul style="margin: 0; padding-left: 20px; color: #ddd; font-size: 14px; line-height: 1.6;">
            <li><strong style="color: #fff;">[Action 1]:</strong> [Specific advice based on {data.get('Temperature')}C, {data.get('Rainfall')}mm rain, or {data.get('pH')} pH]</li>
            <li><strong style="color: #fff;">[Action 2]:</strong> [Specific advice]</li>
            <li><strong style="color: #fff;">[Action 3]:</strong> [Specific advice]</li>
        </ul>
    </div>

    <!-- Risk -->
    <div style="background: rgba(231, 76, 60, 0.1); border-left: 4px solid #e74c3c; padding: 15px; border-radius: 8px;">
        <h4 style="color: #e74c3c; margin: 0 0 10px 0; font-size: 16px;">⚠️ Critical Risk Factor</h4>
        <p style="margin: 0; color: #ddd; font-size: 14px; line-height: 1.5;">[1-2 sentences describing the biggest weather/soil risk to watch out for, fully explained]</p>
    </div>
</div>

ONLY output the HTML block. Do not truncate the HTML output. Ensure all HTML tags are properly closed."""



        print(f"📡 Calling Gemini API... (prompt length: {len(prompt)} chars)")
        
        response = gemini_model.generate_content(prompt)
        
        print(f"📬 Response received. Type: {type(response)}")
        
        # Check for blocked response
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            print(f"⚠️ Prompt feedback: {response.prompt_feedback}")
        
        if not response.parts:
            print("❌ FAIL: response.parts is empty (response blocked or empty)")
            print('='*50)
            return None
        
        result = response.text.strip()
        print(f"✅ Got text response: {len(result)} chars")
        
        # Strip code fences if any
        if result.startswith('```html'):
            result = result[7:]
        elif result.startswith('```'):
            result = result[3:]
        if result.endswith('```'):
            result = result[:-3]
        
        print(f"✅ AI Insight generated successfully!")
        print('='*50)
        return result.strip()
    
    except Exception as e:
        print(f"❌ AI Insight Exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print('='*50)
        return None


def get_bulk_ai_summary(stats):
    """Generate actionable AI farming suggestions based on data analysis"""
    if not gemini_model:
        print("⚠️ Gemini model not configured — skipping AI summary")
        return None
    
    try:
        prompt = f"""You are an expert agricultural advisor. Based on this ML analysis, provide ACTIONABLE RECOMMENDATIONS.

📊 DATASET ANALYSIS:
- Total Records: {stats['total_rows']}
- Total Predicted Yield: {stats['total_yield']} T/Ha (sum)
- Average Yield: {stats['avg_yield']} T/Ha
- Maximum Yield: {stats.get('max_yield', 'N/A')} T/Ha
- Minimum Yield: {stats.get('min_yield', 'N/A')} T/Ha
- Top Performing State: {stats['top_state']}
- Best Crop: {stats['top_crop']}
- High Yield Records (>3 T/Ha): {stats.get('high_yield_count', 0)}
- Low Yield Records (<1 T/Ha): {stats.get('low_yield_count', 0)}

Output ONLY raw HTML. Focus on ACTIONABLE SUGGESTIONS:

<div class="ai-suggestions-container">
    <div class="suggestion-card priority-high">
        <h4>🎯 Priority Actions</h4>
        <ul>
            <li><strong>Action:</strong> [Specific action]</li>
        </ul>
    </div>
    <div class="suggestion-card improvement">
        <h4>📈 Yield Improvement Strategies</h4>
        <ul>
            <li>[Strategy]</li>
        </ul>
    </div>
    <div class="suggestion-card risk">
        <h4>⚠️ Risk Mitigation</h4>
        <ul>
            <li>[Risk and mitigation]</li>
        </ul>
    </div>
</div>

Rules:
- NO description of the data
- Focus on WHAT TO DO
- Be specific with crop names and regions"""
        
        response = gemini_model.generate_content(prompt)
        
        # Guard against blocked/empty responses
        if not response or not response.parts:
            print("⚠️ Gemini returned an empty/blocked response for bulk AI summary")
            return None
        
        result = response.text.strip()
        if result.startswith('```'):
            result = result.split('\n', 1)[1] if '\n' in result else result[3:]
        if result.endswith('```'):
            result = result[:-3]
        return result.strip()
    
    except Exception as e:
        print(f"❌ Bulk AI Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_crop_recommendation_insight(recommended_crop, input_data):
    """Get AI insights for crop recommendation"""
    print(f"\n{'='*50}")
    print(f"🤖 AI RECOMMENDATION REQUEST: {recommended_crop}")
    
    if not gemini_model:
        print("❌ SKIP: gemini_model is None — GOOGLE_API_KEY missing")
        print('='*50)
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

Provide a highly structured, visually appealing agronomic analysis as raw HTML only (no markdown, no code fences).
Use this EXACT HTML structure with inline styles for beautiful rendering:

<div style="display: flex; flex-direction: column; gap: 15px;">
    <!-- Why this crop -->
    <div style="background: rgba(155, 89, 182, 0.1); border-left: 4px solid #9b59b6; padding: 15px; border-radius: 8px;">
        <h4 style="color: #9b59b6; margin: 0 0 10px 0; font-size: 16px;">🌟 Why {recommended_crop}?</h4>
        <p style="margin: 0; color: #ddd; font-size: 14px; line-height: 1.5;">[1-2 sentences explaining why this crop perfectly matches the given NPK, pH, or weather conditions]</p>
    </div>

    <!-- Cultivation Tips -->
    <div style="background: rgba(46, 204, 113, 0.1); border-left: 4px solid #2ecc71; padding: 15px; border-radius: 8px;">
        <h4 style="color: #2ecc71; margin: 0 0 10px 0; font-size: 16px;">🚜 Top 3 Cultivation Tips</h4>
        <ul style="margin: 0; padding-left: 20px; color: #ddd; font-size: 14px; line-height: 1.6;">
            <li><strong style="color: #fff;">[Action 1]:</strong> [Specific advice based on the exact soil/weather data provided]</li>
            <li><strong style="color: #fff;">[Action 2]:</strong> [Specific advice]</li>
            <li><strong style="color: #fff;">[Action 3]:</strong> [Specific advice]</li>
        </ul>
    </div>
</div>

ONLY output the HTML block. Do not truncate the HTML output. Ensure all HTML tags are properly closed."""
        
        print(f"📡 Calling Gemini API... (prompt length: {len(prompt)} chars)")
        response = gemini_model.generate_content(prompt)
        
        if not response or not response.parts:
            print("❌ FAIL: response.parts is empty (response blocked or empty)")
            print('='*50)
            return None
        
        result = response.text.strip()
        print(f"✅ Got text response: {len(result)} chars")
        
        # Strip code fences if any
        if result.startswith('```html'):
            result = result[7:]
        elif result.startswith('```'):
            result = result[3:]
        if result.endswith('```'):
            result = result[:-3]
        
        print(f"✅ AI Insight generated successfully!")
        print('='*50)
        return result.strip()
    
    except Exception as e:
        print(f"❌ Recommendation AI Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print('='*50)
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
        'ensemble_model': ensemble_model is not None,
        'recommend_model': recommend_model is not None,
        'ai_model': gemini_model is not None,
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
    
    if os.path.exists('models/ensemble_metadata.pkl'):
        try:
            info['ensemble_metadata'] = joblib.load('models/ensemble_metadata.pkl')
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
    """Yield prediction with YieldMax Ensemble + AI insights"""
    prediction = None
    ai_insight = None
    error = None
    confidence = None
    prediction_range = None
    individual_predictions = None
    model_weights = None
    model_agreement = None
    data = None
    
    # Check if technical mode requested
    show_technical = request.args.get('technical', 'false').lower() == 'true'
    
    if request.method == 'POST':
        try:
            active_model = ensemble_model if ensemble_model else yield_model
            if not active_model:
                error = "⚠️ No prediction model loaded. Please train the model first."
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
            season = request.form.get('Season', 'Kharif')
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
                    'is_estimated': True
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
            
            # ── Feature Engineering (must match training-time features) ──
            if 'Rainfall' in input_df.columns:
                input_df['Rainfall_Log'] = np.log1p(input_df['Rainfall'])
            if 'Temperature' in input_df.columns and 'Humidity' in input_df.columns:
                input_df['Temp_Humidity_Interaction'] = input_df['Temperature'] * input_df['Humidity']
            if 'Temperature' in input_df.columns:
                input_df['Temperature_Squared'] = input_df['Temperature'] ** 2
            if 'pH' in input_df.columns and 'Rainfall' in input_df.columns:
                input_df['pH_Rainfall_Interaction'] = input_df['pH'] * input_df['Rainfall']
            
            for f in feature_names:
                if f not in input_df.columns:
                    input_df[f] = 0
            
            X_pred = input_df[feature_names]
            
            # Use ensemble if available, fallback to single model
            if ensemble_model:
                result = ensemble_model.predict(X_pred, return_details=True)
                pred_val = max(0, float(result['final_prediction'][0]))
                confidence = round(float(result['confidence'][0]), 1)
                prediction_range = {
                    'lower': round(max(0, float(result['prediction_interval']['lower'][0])), 2),
                    'upper': round(float(result['prediction_interval']['upper'][0]), 2)
                }
                
                if show_technical:
                    individual_predictions = {
                        'xgboost': round(max(0, float(result['individual_predictions']['xgboost'][0])), 2),
                        'lightgbm': round(max(0, float(result['individual_predictions']['lightgbm'][0])), 2),
                        'neural_network': round(max(0, float(result['individual_predictions']['neural_network'][0])), 2)
                    }
                    model_weights = result['model_weights']
                    model_agreement = round(result['model_agreement'], 1)
            else:
                pred_val = max(0, float(yield_model.predict(X_pred)[0]))
                confidence = None
            
            prediction = f"{pred_val:.2f}"
            
            ai_insight = get_ai_insight(data, prediction, confidence, prediction_range)
            save_prediction_history(data, prediction, 'yield')
            
        except ValueError as ve:
            error = f"❌ Invalid input format: {str(ve)}"
        except Exception as e:
            error = f"❌ Prediction error: {str(e)}"
    
    return render_template('predict_yield.html', 
                         prediction=prediction, 
                         ai_insight=ai_insight,
                         confidence=confidence,
                         prediction_range=prediction_range,
                         show_technical=show_technical,
                         individual_predictions=individual_predictions,
                         model_weights=model_weights,
                         model_agreement=model_agreement,
                         input_data=data if prediction else None,
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
        
        # ── Fill missing environmental columns with realistic agricultural defaults ──
        # Defaulting to 0 causes the model to predict wildly wrong values
        ENV_DEFAULTS = {
            'Temperature': 25.0,   # typical Indian avg temp °C
            'Humidity':    65.0,   # typical relative humidity %
            'pH':           6.5,   # neutral-ish soil pH
            'Rainfall':  1000.0,   # typical annual rainfall mm
            'Crop_Year':  2020.0,
            'Area':          1.0,
        }
        for col, default_val in ENV_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default_val
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default_val)
        
        # ── Feature Engineering (must match training-time features) ──
        if 'Rainfall' in df.columns:
            df['Rainfall_Log'] = np.log1p(df['Rainfall'])
        if 'Temperature' in df.columns and 'Humidity' in df.columns:
            df['Temp_Humidity_Interaction'] = df['Temperature'] * df['Humidity']
        if 'Temperature' in df.columns:
            df['Temperature_Squared'] = df['Temperature'] ** 2
        if 'pH' in df.columns and 'Rainfall' in df.columns:
            df['pH_Rainfall_Interaction'] = df['pH'] * df['Rainfall']
        
        for f in feature_names:
            if f not in df.columns:
                df[f] = 0
        
        X = df[feature_names]
        
        # Use ensemble if available, fallback to single model
        if ensemble_model:
            result = ensemble_model.predict(X.values, return_details=False)
            predictions = result[0]  # (predictions, confidence)
        elif yield_model:
            predictions = yield_model.predict(X)
        else:
            return "❌ No prediction model loaded", 500
        
        # Clip predictions: yield T/Ha can never be negative
        original_df['Predicted_Yield_THa'] = np.maximum(predictions, 0).round(4)
        # Also rename old column if it exists
        original_df.rename(columns={'Predicted_Yield_tonnes': 'Predicted_Yield_THa'}, errors='ignore', inplace=True)
        
        output_path = os.path.join('static', 'predicted_yield.csv')
        os.makedirs('static', exist_ok=True)
        original_df.to_csv(output_path, index=False)
        
        # ANALYTICS — thresholds now in T/Ha scale (not raw Production tonnes)
        total_yield = round(original_df['Predicted_Yield_THa'].sum(), 4)
        avg_yield   = round(original_df['Predicted_Yield_THa'].mean(), 4)
        max_yield   = round(original_df['Predicted_Yield_THa'].max(), 4)
        min_yield   = round(original_df['Predicted_Yield_THa'].min(), 4)
        std_yield   = round(original_df['Predicted_Yield_THa'].std(), 4)
        total_rows  = len(original_df)
        
        # T/Ha Yield Distribution  (0-1 | 1-2 | 2-3 | 3-5 | 5+)
        yield_bins   = [0, 1, 2, 3, 5, float('inf')]
        yield_labels = ['0-1', '1-2', '2-3', '3-5', '5+']
        yield_distribution = pd.cut(original_df['Predicted_Yield_THa'], bins=yield_bins, labels=yield_labels).value_counts().sort_index()
        yield_dist_data = yield_distribution.values.tolist()
        
        # High/Medium/Low yield in T/Ha
        high_yield_count   = len(original_df[original_df['Predicted_Yield_THa'] > 3])
        low_yield_count    = len(original_df[original_df['Predicted_Yield_THa'] < 1])
        medium_yield_count = total_rows - high_yield_count - low_yield_count
        
        confidence_ratio = 1 - (std_yield / (abs(avg_yield) + 1e-6))
        model_confidence = max(0.6, min(0.95, abs(confidence_ratio)))
        
        # Feature Importance
        feature_importance = {}
        try:
            if hasattr(yield_model, 'feature_importances_'):
                importances = yield_model.feature_importances_
                feature_names_loaded = joblib.load('models/yield_features.pkl')
                top_indices = np.argsort(importances)[::-1][:5]
                for idx in top_indices:
                    feature_importance[feature_names_loaded[idx]] = round(float(importances[idx]) * 100, 1)
        except Exception as e:
            print(f"Feature importance error: {e}")
            feature_importance = {
                'Rainfall': 28.5, 'Temperature': 22.3, 'State': 18.7, 
                'Crop': 15.2, 'pH': 10.1
            }
        
        # State analysis
        state_col = None
        for col in original_df.columns:
            if 'state' in col.lower() and 'name' in col.lower():
                state_col = col
                break
        if not state_col:
            state_col = next((col for col in original_df.columns if 'state' in col.lower() and 'code' not in col.lower()), None)
        
        state_labels, state_yields, top_state, all_states_count = [], [], "N/A", 0
        
        if state_col:
            state_grp = original_df.groupby(state_col)['Predicted_Yield_THa'].mean()
            all_states_count = len(state_grp)
            state_grp = state_grp.sort_values(ascending=False).head(5)
            top_state = state_grp.index[0]
            state_labels = state_grp.index.tolist()
            state_yields = [round(y, 2) for y in state_grp.values]
        
        # Crop analysis
        crop_col = next((col for col in original_df.columns 
                        if 'crop' in col.lower() and 'year' not in col.lower()), None)
        crop_labels, crop_counts, crop_yields, top_crop, all_crops_count = [], [], [], "N/A", 0
        
        if crop_col:
            crop_grp = original_df.groupby(crop_col)['Predicted_Yield_THa'].agg(['mean', 'count'])
            all_crops_count = len(crop_grp)
            top_crop = crop_grp['mean'].idxmax()
            crop_grp = crop_grp.nlargest(5, 'count')
            crop_labels = crop_grp.index.tolist()
            crop_counts = crop_grp['count'].values.tolist()
            crop_yields = [round(y, 2) for y in crop_grp['mean'].values]
        
        preview_cols = original_df.columns.tolist()
        preview_data = original_df.head(10).values.tolist()
        
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
            total_rows=total_rows,
            total_yield=f"{total_yield:,.2f}",
            avg_yield=avg_yield,
            max_yield=max_yield,
            min_yield=min_yield,
            top_state=top_state,
            top_crop=top_crop,
            model_confidence=round(model_confidence * 100, 1),
            high_yield_count=high_yield_count,
            low_yield_count=low_yield_count,
            medium_yield_count=medium_yield_count,
            yield_dist_labels=yield_labels,
            yield_dist_data=yield_dist_data,
            all_states_count=all_states_count,
            all_crops_count=all_crops_count,
            feature_importance=feature_importance,
            state_labels=state_labels,
            state_yields=state_yields,
            crop_labels=crop_labels,
            crop_counts=crop_counts,
            crop_yields=crop_yields,
            columns=preview_cols,
            preview_data=preview_data,
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
            
            pred_id = int(recommend_model.predict(final_features)[0])
            if recommend_classes and len(recommend_classes) > pred_id:
                recommendation = str(recommend_classes[pred_id]).title()
            else:
                recommendation = f"Crop {pred_id}"
            
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
    """Smart Crop Recommendation Wizard"""
    
    states = list(REGIONAL_DATA.get('states', {}).keys()) if REGIONAL_DATA else []
    soil_types = list(REGIONAL_DATA.get('soil_npk_profiles', {}).keys()) if REGIONAL_DATA else ['loamy', 'sandy', 'clay', 'black', 'red']
    
    if request.method == 'POST':
        try:
            user_inputs = {
                'state': request.form.get('state', ''),
                'district': request.form.get('district', ''),
                'season': request.form.get('season', 'kharif'),
                'soil_type': request.form.get('soil_type', 'loamy'),
                'previous_crop': request.form.get('previous_crop', 'none'),
                'water_source': request.form.get('water_source', 'rainfed')
            }
            
            is_advanced = request.form.get('mode') == 'advanced'
            
            if is_advanced:
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
                estimated = estimate_npk_from_inputs(
                    user_inputs['state'],
                    user_inputs['soil_type'],
                    user_inputs['season'],
                    user_inputs['previous_crop'],
                    user_inputs['water_source']
                )
            
            recommendations = get_top_crop_recommendations(estimated, top_n=3)
            
            if not recommendations:
                return render_template('recommend.html',
                    states=states,
                    soil_types=soil_types,
                    error="⚠️ Could not generate recommendations. Model may not be loaded.")
            
            ai_insight = get_wizard_ai_insight(recommendations, user_inputs, estimated)
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


@app.route('/api/estimate_conditions')
def api_estimate_conditions():
    """API endpoint to get estimated environmental conditions for smart defaults"""
    state = request.args.get('state', '')
    district = request.args.get('district', '')
    season = request.args.get('season', 'Kharif')
    
    # Use the existing estimation function
    estimated = estimate_yield_conditions(state, district, season)
    
    return jsonify(estimated)


# ========================================
# ROUTES - MODEL ANALYTICS
# ========================================

@app.route('/model_dashboard')
def model_dashboard():
    """Model Performance Dashboard"""
    model_info = get_model_info()
    
    training_logs = []
    logs_dir = 'logs'
    if os.path.exists(logs_dir):
        log_files = sorted([f for f in os.listdir(logs_dir) if f.startswith('training_log')], reverse=True)
        training_logs = log_files[:5]
    
    return render_template('model_dashboard.html', 
                         model_info=model_info, 
                         training_logs=training_logs)


@app.route('/model_comparison')
def model_comparison():
    """Model Comparison"""
    model_info = get_model_info()
    return render_template('model_comparison.html', model_info=model_info)


# ========================================
# ROUTES - STATIC PAGES
# ========================================

@app.route('/documentation')
def documentation():
    """Documentation Page"""
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
    
    print("\n" + "="*55)
    print("🌾 AgriVision - YieldMax Precision Model")
    print("   Advanced Multi-Algorithm Intelligence")
    print("="*55)
    print(f"  ✅ Ensemble Model : {'Loaded' if ensemble_model else 'Not Trained Yet'}")
    print(f"  ✅ Yield Model    : {'Loaded' if yield_model else 'Not Found'}")
    print(f"  ✅ Recommend Model: {'Loaded' if recommend_model else 'Not Found'}")
    print(f"  ✅ AI Model       : {'Configured' if gemini_model else 'Not Configured'}")
    print("="*55 + "\n")
    
    app.run(
        debug=os.getenv('FLASK_DEBUG', 'false').lower() == 'true',
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        threaded=True
    )
