"""
🌿 AI Plant Doctor - Disease Detection Module
Part of AgriVision

Uses a CNN (MobileNetV2) to detect plant diseases from leaf images.
"""

import os
import numpy as np
from PIL import Image
import io
import json
from datetime import datetime

# ========================================
# TENSORFLOW CONFIGURATION
# ========================================

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KERAS_BACKEND'] = 'tensorflow'

try:
    import tensorflow as tf
    
    # Memory optimization - only allocate GPU memory as needed
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    import keras
    from keras.models import load_model
    from keras.utils import img_to_array
    from keras.applications.mobilenet_v2 import preprocess_input
    
    TF_AVAILABLE = True
    print(f"✅ TensorFlow {tf.__version__} loaded")
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not installed - using mock mode")


# ========================================
# DISEASE DATABASE
# ========================================

DISEASE_DATABASE = {}
CACHE_PATH = 'models/disease_cache.json'

if os.path.exists(CACHE_PATH):
    try:
        with open(CACHE_PATH, 'r') as f:
            DISEASE_DATABASE = json.load(f)
        print(f"✅ Loaded {len(DISEASE_DATABASE)} diseases from cache")
    except Exception as e:
        print(f"⚠️ Failed to load disease cache: {e}")


# ========================================
# PLANT DOCTOR CLASS
# ========================================

class PlantDoctor:
    """
    CNN-based plant disease detection using MobileNetV2.
    
    Attributes:
        model_path: Path to the trained .h5 model
        model: Loaded Keras model
        img_size: Input image dimensions (224x224 for MobileNetV2)
        class_names: List of disease class names
    """
    
    def __init__(self, model_path='models/disease_model.h5'):
        """Initialize PlantDoctor with model path."""
        self.model_path = model_path
        self.model = None
        self.img_size = (224, 224)
        self.class_names = []
        
        self._load_class_names()
        self._load_model()
    
    def _load_class_names(self):
        """Load class names from JSON file."""
        json_path = os.path.join(os.path.dirname(self.model_path), 'classes.json')
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    self.class_names = json.load(f)
                print(f"✅ Loaded {len(self.class_names)} disease classes")
            except Exception as e:
                print(f"⚠️ Failed to load classes.json: {e}")
                self.class_names = list(DISEASE_DATABASE.keys())
        else:
            print("⚠️ classes.json not found")
            self.class_names = list(DISEASE_DATABASE.keys())
    
    def _load_model(self):
        """Load the trained CNN model."""
        if not TF_AVAILABLE:
            print("⚠️ TensorFlow not available - using mock predictions")
            return
        
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                print(f"✅ Disease model loaded: {self.model_path}")
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                self.model = None
        else:
            print(f"⚠️ Model not found: {self.model_path}")
    
    def preprocess_image(self, img_input):
        """
        Preprocess image for model prediction.
        
        Args:
            img_input: File path (str), bytes, or PIL Image
        
        Returns:
            Preprocessed numpy array with shape (1, 224, 224, 3)
        """
        # Handle different input types
        if isinstance(img_input, str):
            img = Image.open(img_input)
        elif isinstance(img_input, bytes):
            img = Image.open(io.BytesIO(img_input))
        elif hasattr(img_input, 'read'):
            img = Image.open(img_input)
        else:
            img = img_input
        
        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize(self.img_size)
        
        # Convert to array
        if TF_AVAILABLE:
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
        else:
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, img_input):
        """
        Predict disease from leaf image.
        
        Args:
            img_input: Image as file path, bytes, or PIL Image
        
        Returns:
            dict with prediction results including:
            - success: bool
            - prediction: disease info
            - diagnosis: symptoms and cause
            - treatment: chemical and organic options
            - top_predictions: top 3 predictions
        """
        try:
            processed_img = self.preprocess_image(img_input)
            
            if self.model is not None and TF_AVAILABLE:
                # Real CNN prediction
                predictions = self.model.predict(processed_img, verbose=0)
                predicted_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_idx])
                
                # Top 3 predictions
                top_indices = np.argsort(predictions[0])[-3:][::-1]
                top_predictions = [
                    {
                        "disease": self.class_names[i] if i < len(self.class_names) else f"Class_{i}",
                        "confidence": float(predictions[0][i])
                    }
                    for i in top_indices
                ]
            else:
                # Mock prediction for testing
                if len(self.class_names) > 0:
                    predicted_idx = np.random.choice(len(self.class_names))
                else:
                    predicted_idx = 0
                confidence = np.random.uniform(0.75, 0.98)
                top_predictions = [
                    {"disease": self.class_names[predicted_idx] if self.class_names else "Unknown", "confidence": confidence}
                ]
            
            # Get disease key
            disease_key = self.class_names[predicted_idx] if predicted_idx < len(self.class_names) else "Unknown"
            disease_info = DISEASE_DATABASE.get(disease_key, {})
            
            return {
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
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_supported_diseases(self):
        """Get list of all detectable diseases."""
        return [
            {
                "key": key,
                "name": info.get("name", key),
                "crop": info.get("crop", "Unknown"),
                "severity": info.get("severity", "Unknown")
            }
            for key, info in DISEASE_DATABASE.items()
        ]
    
    def get_supported_crops(self):
        """Get unique list of supported crops."""
        crops = set(info.get("crop", "") for info in DISEASE_DATABASE.values())
        return sorted([c for c in crops if c])


# ========================================
# SINGLETON INSTANCE
# ========================================

_plant_doctor = None

def get_plant_doctor():
    """Get or create PlantDoctor singleton instance."""
    global _plant_doctor
    if _plant_doctor is None:
        _plant_doctor = PlantDoctor()
    return _plant_doctor


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
    print(f"🤖 Model Status: {'Loaded' if doctor.model else 'Mock Mode'}")
