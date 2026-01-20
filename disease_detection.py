"""
🌿 AI Plant Doctor - Disease Detection Module
Part of AgriVision v3.0

Uses a CNN (MobileNetV2) to detect plant diseases from leaf images.
"""

import os
import numpy as np
from PIL import Image
import io
import json
from datetime import datetime

# TensorFlow imports (with graceful fallback)
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image as keras_image
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not installed. Disease detection will use mock mode.")


# ========================================
# DISEASE DATABASE
# ========================================

DISEASE_DATABASE = {
    "Tomato_Early_Blight": {
        "name": "Early Blight",
        "crop": "Tomato",
        "severity": "Moderate",
        "symptoms": "Dark brown spots with concentric rings on lower leaves",
        "cause": "Fungus (Alternaria solani)",
        "treatment": {
            "chemical": "Apply Mancozeb or Chlorothalonil fungicide every 7-10 days",
            "organic": "Remove infected leaves, apply neem oil spray, ensure proper spacing"
        },
        "prevention": "Crop rotation, avoid overhead watering, mulching"
    },
    "Tomato_Late_Blight": {
        "name": "Late Blight",
        "crop": "Tomato",
        "severity": "Severe",
        "symptoms": "Water-soaked spots, white mold on leaf undersides",
        "cause": "Oomycete (Phytophthora infestans)",
        "treatment": {
            "chemical": "Copper-based fungicides or Metalaxyl immediately",
            "organic": "Remove and destroy all infected plants, do not compost"
        },
        "prevention": "Resistant varieties, good air circulation, avoid wet foliage"
    },
    "Tomato_Leaf_Mold": {
        "name": "Leaf Mold",
        "crop": "Tomato",
        "severity": "Moderate",
        "symptoms": "Yellow patches on upper leaf, olive-green mold below",
        "cause": "Fungus (Passalora fulva)",
        "treatment": {
            "chemical": "Apply fungicides with chlorothalonil",
            "organic": "Improve ventilation, reduce humidity, prune lower leaves"
        },
        "prevention": "Greenhouse ventilation, resistant varieties"
    },
    "Tomato_Healthy": {
        "name": "Healthy",
        "crop": "Tomato",
        "severity": "None",
        "symptoms": "No disease symptoms detected",
        "cause": "N/A",
        "treatment": {
            "chemical": "No treatment needed",
            "organic": "Continue regular care"
        },
        "prevention": "Maintain current practices"
    },
    "Potato_Early_Blight": {
        "name": "Early Blight",
        "crop": "Potato",
        "severity": "Moderate",
        "symptoms": "Brown spots with target-like rings, yellowing leaves",
        "cause": "Fungus (Alternaria solani)",
        "treatment": {
            "chemical": "Apply Mancozeb or Azoxystrobin",
            "organic": "Remove infected foliage, apply copper spray"
        },
        "prevention": "Certified seed potatoes, proper spacing"
    },
    "Potato_Late_Blight": {
        "name": "Late Blight",
        "crop": "Potato",
        "severity": "Critical",
        "symptoms": "Dark water-soaked lesions, white fuzzy growth",
        "cause": "Oomycete (Phytophthora infestans)",
        "treatment": {
            "chemical": "Metalaxyl-M + Mancozeb immediately",
            "organic": "Destroy all infected plants, copper fungicide"
        },
        "prevention": "Resistant varieties, avoid overhead irrigation"
    },
    "Potato_Healthy": {
        "name": "Healthy",
        "crop": "Potato",
        "severity": "None",
        "symptoms": "No disease symptoms detected",
        "cause": "N/A",
        "treatment": {
            "chemical": "No treatment needed",
            "organic": "Continue regular care"
        },
        "prevention": "Maintain current practices"
    },
    "Corn_Common_Rust": {
        "name": "Common Rust",
        "crop": "Corn/Maize",
        "severity": "Moderate",
        "symptoms": "Reddish-brown pustules on both leaf surfaces",
        "cause": "Fungus (Puccinia sorghi)",
        "treatment": {
            "chemical": "Foliar fungicides (Triazoles, Strobilurins)",
            "organic": "Plant resistant hybrids, remove infected debris"
        },
        "prevention": "Early planting, resistant varieties"
    },
    "Corn_Gray_Leaf_Spot": {
        "name": "Gray Leaf Spot",
        "crop": "Corn/Maize",
        "severity": "Severe",
        "symptoms": "Rectangular gray-brown lesions parallel to leaf veins",
        "cause": "Fungus (Cercospora zeae-maydis)",
        "treatment": {
            "chemical": "Strobilurin or Triazole fungicides at tasseling",
            "organic": "Crop rotation, tillage to bury residue"
        },
        "prevention": "Resistant hybrids, residue management"
    },
    "Corn_Healthy": {
        "name": "Healthy",
        "crop": "Corn/Maize",
        "severity": "None",
        "symptoms": "No disease symptoms detected",
        "cause": "N/A",
        "treatment": {
            "chemical": "No treatment needed",
            "organic": "Continue regular care"
        },
        "prevention": "Maintain current practices"
    },
    "Rice_Brown_Spot": {
        "name": "Brown Spot",
        "crop": "Rice",
        "severity": "Moderate",
        "symptoms": "Oval brown spots with gray centers on leaves",
        "cause": "Fungus (Bipolaris oryzae)",
        "treatment": {
            "chemical": "Propiconazole or Tricyclazole spray",
            "organic": "Balanced fertilization, seed treatment with Trichoderma"
        },
        "prevention": "Use clean certified seeds, balanced NPK"
    },
    "Rice_Leaf_Blast": {
        "name": "Leaf Blast",
        "crop": "Rice",
        "severity": "Severe",
        "symptoms": "Diamond-shaped lesions with gray centers",
        "cause": "Fungus (Magnaporthe oryzae)",
        "treatment": {
            "chemical": "Tricyclazole or Isoprothiolane spray",
            "organic": "Silicon fertilization, resistant varieties"
        },
        "prevention": "Avoid excess nitrogen, use resistant varieties"
    },
    "Rice_Healthy": {
        "name": "Healthy",
        "crop": "Rice",
        "severity": "None",
        "symptoms": "No disease symptoms detected",
        "cause": "N/A",
        "treatment": {
            "chemical": "No treatment needed",
            "organic": "Continue regular care"
        },
        "prevention": "Maintain current practices"
    },
    "Wheat_Leaf_Rust": {
        "name": "Leaf Rust",
        "crop": "Wheat",
        "severity": "Moderate",
        "symptoms": "Orange-brown pustules scattered on leaves",
        "cause": "Fungus (Puccinia triticina)",
        "treatment": {
            "chemical": "Propiconazole or Tebuconazole spray",
            "organic": "Resistant varieties, early sowing"
        },
        "prevention": "Resistant varieties, eliminate volunteer wheat"
    },
    "Wheat_Healthy": {
        "name": "Healthy",
        "crop": "Wheat",
        "severity": "None",
        "symptoms": "No disease symptoms detected",
        "cause": "N/A",
        "treatment": {
            "chemical": "No treatment needed",
            "organic": "Continue regular care"
        },
        "prevention": "Maintain current practices"
    }
}

# Class names matching PlantVillage dataset style
CLASS_NAMES = list(DISEASE_DATABASE.keys())


# ========================================
# DISEASE DETECTION CLASS
# ========================================

class PlantDoctor:
    """
    AI-powered plant disease detection using CNN
    """
    
    def __init__(self, model_path='models/disease_model.h5'):
        self.model_path = model_path
        self.model = None
        self.img_size = (224, 224)  # MobileNetV2 default
        self.class_names = []
        self._load_class_names()
        
        # Load model if available
        self._load_model()
    
    def _load_class_names(self):
        """Load class names from JSON or fallback to database"""
        json_path = os.path.join(os.path.dirname(self.model_path), 'classes.json')
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    self.class_names = json.load(f)
                print(f"✅ Loaded {len(self.class_names)} classes from {json_path}")
            except Exception as e:
                print(f"⚠️ Failed to load classes.json: {e}")
                self.class_names = CLASS_NAMES
        else:
            print("⚠️ classes.json not found. Using default database keys (may be mismatched!)")
            self.class_names = CLASS_NAMES
    
    def _load_model(self):
        """Load the trained CNN model"""
        if not TF_AVAILABLE:
            print("⚠️ TensorFlow not available - using mock predictions")
            return
        
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                print(f"✅ Disease detection model loaded from {self.model_path}")
            except Exception as e:
                print(f"❌ Failed to load disease model: {e}")
                self.model = None
        else:
            print(f"⚠️ Disease model not found at {self.model_path}")
            print("   Run train_disease_model.py to train the model")
    
    def preprocess_image(self, img_input):
        """
        Preprocess image for model prediction
        
        Args:
            img_input: Can be file path, PIL Image, or bytes
        
        Returns:
            Preprocessed numpy array ready for model
        """
        # Handle different input types
        if isinstance(img_input, str):
            # File path
            img = Image.open(img_input)
        elif isinstance(img_input, bytes):
            # Raw bytes
            img = Image.open(io.BytesIO(img_input))
        elif hasattr(img_input, 'read'):
            # File-like object
            img = Image.open(img_input)
        else:
            # Assume PIL Image
            img = img_input
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to model input size
        img = img.resize(self.img_size)
        
        # Convert to array and preprocess
        img_array = keras_image.img_to_array(img) if TF_AVAILABLE else np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        if TF_AVAILABLE:
            img_array = preprocess_input(img_array)
        else:
            img_array = img_array / 255.0
        
        return img_array
    
    def predict(self, img_input):
        """
        Predict disease from image
        
        Args:
            img_input: Image file path, bytes, or PIL Image
        
        Returns:
            dict with prediction results
        """
        try:
            # Preprocess image
            processed_img = self.preprocess_image(img_input)
            
            if self.model is not None and TF_AVAILABLE:
                # Real prediction
                predictions = self.model.predict(processed_img, verbose=0)
                predicted_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_idx])
                
                # Get top 3 predictions
                top_indices = np.argsort(predictions[0])[-3:][::-1]
                top_predictions = [
                    {
                        "disease": self.class_names[i],
                        "confidence": float(predictions[0][i])
                    }
                    for i in top_indices
                ]
            else:
                # Mock prediction for demo/testing
                predicted_idx = np.random.choice(len(self.class_names))
                confidence = np.random.uniform(0.75, 0.98)
                top_predictions = [
                    {"disease": self.class_names[predicted_idx], "confidence": confidence},
                    {"disease": self.class_names[(predicted_idx + 1) % len(self.class_names)], "confidence": 0.1},
                    {"disease": self.class_names[(predicted_idx + 2) % len(self.class_names)], "confidence": 0.05}
                ]
            
            # Get disease info
            disease_key = self.class_names[predicted_idx]
            disease_info = DISEASE_DATABASE.get(disease_key, {})
            
            result = {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "prediction": {
                    "disease_key": disease_key,
                    "disease_name": disease_info.get("name", "Unknown"),
                    "crop": disease_info.get("crop", "Unknown"),
                    "confidence": round(confidence * 100, 2),
                    "severity": disease_info.get("severity", "Unknown")
                },
                "diagnosis": {
                    "symptoms": disease_info.get("symptoms", ""),
                    "cause": disease_info.get("cause", "")
                },
                "treatment": disease_info.get("treatment", {}),
                "prevention": disease_info.get("prevention", ""),
                "top_predictions": top_predictions,
                "model_used": "CNN (MobileNetV2)" if self.model else "Mock Mode"
            }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_supported_diseases(self):
        """Get list of all detectable diseases"""
        return [
            {
                "key": key,
                "name": info["name"],
                "crop": info["crop"],
                "severity": info["severity"]
            }
            for key, info in DISEASE_DATABASE.items()
        ]
    
    def get_supported_crops(self):
        """Get unique list of supported crops"""
        crops = set(info["crop"] for info in DISEASE_DATABASE.values())
        return sorted(list(crops))


# ========================================
# SINGLETON INSTANCE
# ========================================

# Global instance for Flask app
plant_doctor = None

def get_plant_doctor():
    """Get or create PlantDoctor instance"""
    global plant_doctor
    if plant_doctor is None:
        plant_doctor = PlantDoctor()
    return plant_doctor


# ========================================
# CLI TEST
# ========================================

if __name__ == "__main__":
    print("=" * 50)
    print("🌿 AI PLANT DOCTOR - Disease Detection Module")
    print("=" * 50)
    
    doctor = get_plant_doctor()
    
    print(f"\n📋 Supported Crops: {doctor.get_supported_crops()}")
    print(f"📋 Total Diseases: {len(doctor.get_supported_diseases())}")
    
    # Test with mock prediction
    print("\n🧪 Testing mock prediction...")
    result = doctor.predict(np.random.rand(224, 224, 3))
    
    print(f"\n📊 Prediction Result:")
    print(f"   Disease: {result['prediction']['disease_name']}")
    print(f"   Crop: {result['prediction']['crop']}")
    print(f"   Confidence: {result['prediction']['confidence']}%")
    print(f"   Severity: {result['prediction']['severity']}")
    print(f"\n💊 Treatment:")
    print(f"   Chemical: {result['treatment'].get('chemical', 'N/A')}")
    print(f"   Organic: {result['treatment'].get('organic', 'N/A')}")
