"""Quick test to check model loading"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test 1: Check pkl contents
import joblib
print("=== Test 1: PKL file contents ===")
try:
    d = joblib.load('models/yieldmax_ensemble.pkl')
    print(f"Keys: {list(d.keys())}")
    print(f"Has meta_learner: {'meta_learner' in d}")
    print(f"Has ensemble: {'ensemble' in d}")
    print(f"Has dnn_path: {'dnn_path' in d}")
    print(f"Has dnn_model: {'dnn_model' in d}")
except Exception as e:
    print(f"Error loading pkl: {e}")

# Test 2: Check DNN h5 file
print("\n=== Test 2: DNN h5 file ===")
dnn_path = 'models/yieldmax_ensemble_dnn.h5'
print(f"DNN file exists: {os.path.exists(dnn_path)}")

# Test 3: Try full load
print("\n=== Test 3: Full model load ===")
try:
    from ensemble_model import YieldMaxEnsemble
    m = YieldMaxEnsemble()
    m.load('models/yieldmax_ensemble.pkl')
    print(f"meta_learner loaded: {m.meta_learner is not None}")
    print(f"xgb_model loaded: {m.xgb_model is not None}")
    print(f"lgbm_model loaded: {m.lgbm_model is not None}")
    print(f"dnn_model loaded: {m.dnn_model is not None}")
    
    # Test prediction
    import numpy as np
    X_test = np.zeros((1, m.n_features))
    pred, conf = m.predict(X_test)
    print(f"Test prediction: {pred[0]:.4f}, confidence: {conf[0]:.1f}%")
    print("SUCCESS: Model loads and predicts correctly!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check label encoders
print("\n=== Test 4: Label encoders ===")
try:
    le = joblib.load('models/yield_label_encoders.pkl')
    print(f"Type: {type(le)}")
    print(f"Keys: {list(le.keys())}")
except Exception as e:
    print(f"Error: {e}")

# Test 5: Check features
print("\n=== Test 5: Features ===")
try:
    features = joblib.load('models/yield_features.pkl')
    print(f"Features ({len(features)}): {features}")
except Exception as e:
    print(f"Error: {e}")
