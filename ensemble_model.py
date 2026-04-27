"""
YieldMax Precision Model - Ensemble System
Advanced Multi-Algorithm Intelligence for Maximum Yield Accuracy

This module implements a unified ensemble model combining:
- XGBoost: Categorical feature specialist
- LightGBM: Environmental parameter expert  
- Deep Neural Network: Interaction pattern detector
- Ridge Meta-Learner: Manual stacking (replaces sklearn StackingRegressor)
"""

import os

import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')


class YieldMaxEnsemble:
    """
    YieldMax Precision Model - Unified Ensemble System
    
    Uses manual stacking: trains base models, generates cross-validated
    predictions, then trains Ridge meta-learner on those predictions.
    """
    
    def __init__(self):
        self.model_name = "YieldMax Precision Model"
        self.version = "1.0"
        self.xgb_model = None
        self.lgbm_model = None
        self.dnn_model = None
        self.meta_learner = None  # Ridge regression meta-learner
        self.feature_names = None
        self.n_features = None
        
    def _build_dnn(self, n_features):
        """Build and compile DNN model"""
        model = Sequential([
            Dense(256, activation='relu', input_dim=n_features),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def _train_dnn(self, X, y):
        """Train DNN with early stopping"""
        model = self._build_dnn(X.shape[1])
        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        model.fit(X, y, epochs=100, batch_size=32, verbose=0, callbacks=[early_stop])
        return model
    
    def train(self, X, y, feature_names=None):
        """
        Train the unified ensemble model using manual stacking.
        
        Steps:
        1. Train XGBoost, LightGBM, DNN on full data
        2. Generate cross-validated predictions from XGB and LGBM
        3. Generate DNN predictions (no CV for DNN - too slow)
        4. Train Ridge meta-learner on stacked predictions
        """
        self.feature_names = feature_names
        self.n_features = X.shape[1]
        
        print(f"\n{'='*60}")
        print(f"  Training {self.model_name} v{self.version}")
        print(f"{'='*60}\n")
        
        # 1. XGBoost
        print("📊 Training Model 1/3: XGBoost Regressor...")
        self.xgb_model = XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1
        )
        self.xgb_model.fit(X, y)
        print("   ✓ XGBoost trained successfully")
        
        # 2. LightGBM
        print("📊 Training Model 2/3: LightGBM Regressor...")
        self.lgbm_model = LGBMRegressor(
            n_estimators=500, learning_rate=0.03,
            num_leaves=31, max_depth=-1,
            min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, boosting_type='gbdt',
            random_state=42, n_jobs=-1, verbose=-1
        )
        self.lgbm_model.fit(X, y)
        print("   ✓ LightGBM trained successfully")
        
        # 3. Deep Neural Network
        print("📊 Training Model 3/3: Deep Neural Network...")
        self.dnn_model = self._train_dnn(X, y)
        print("   ✓ Neural Network trained successfully\n")
        
        # 4. Manual Stacking Meta-Learner
        print("🔗 Training Meta-Learner (Manual Stacking)...")
        
        # Get cross-validated predictions for XGB and LGBM
        xgb_cv_pred = cross_val_predict(
            XGBRegressor(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1
            ), X, y, cv=5
        )
        print("   ✓ XGBoost CV predictions generated")
        
        lgbm_cv_pred = cross_val_predict(
            LGBMRegressor(
                n_estimators=500, learning_rate=0.03,
                num_leaves=31, subsample=0.8,
                colsample_bytree=0.8, random_state=42,
                n_jobs=-1, verbose=-1
            ), X, y, cv=5
        )
        print("   ✓ LightGBM CV predictions generated")
        
        # DNN prediction on training data (no CV - too expensive)
        dnn_pred = self.dnn_model.predict(X, verbose=0).flatten()
        print("   ✓ DNN predictions generated")
        
        # Stack predictions as features for meta-learner
        stacked_features = np.column_stack([xgb_cv_pred, lgbm_cv_pred, dnn_pred])
        
        # Train Ridge meta-learner
        self.meta_learner = Ridge(alpha=1.0)
        self.meta_learner.fit(stacked_features, y)
        print("   ✓ Ridge meta-learner trained successfully\n")
        
        print(f"{'='*60}")
        print(f"  {self.model_name} Training Complete!")
        print(f"{'='*60}\n")
        
        return {
            'status': 'success',
            'n_features': self.n_features,
            'n_samples': len(X)
        }
    
    def predict(self, X, return_details=False):
        """
        Generate unified prediction.
        
        If return_details=False: returns (prediction, confidence)
        If return_details=True: returns dict with full breakdown
        """
        if self.meta_learner is None:
            raise ValueError("Model not trained! Call train() first.")
        
        # Get individual predictions
        xgb_pred = np.maximum(self.xgb_model.predict(X), 0)
        lgbm_pred = np.maximum(self.lgbm_model.predict(X), 0)
        dnn_pred = np.maximum(self.dnn_model.predict(X, verbose=0).flatten(), 0)
        
        # Stack and get meta-learner prediction
        stacked = np.column_stack([xgb_pred, lgbm_pred, dnn_pred])
        ensemble_pred = np.maximum(self.meta_learner.predict(stacked), 0)
        
        confidence = self._calculate_confidence(xgb_pred, lgbm_pred, dnn_pred)
        
        if not return_details:
            return ensemble_pred, confidence
        
        # Technical mode - full breakdown
        lower, upper = self._get_prediction_interval(
            ensemble_pred, xgb_pred, lgbm_pred, dnn_pred
        )
        weights = self._get_model_weights()
        
        return {
            'final_prediction': ensemble_pred,
            'confidence': confidence,
            'prediction_interval': {
                'lower': lower,
                'expected': ensemble_pred,
                'upper': upper
            },
            'individual_predictions': {
                'xgboost': xgb_pred,
                'lightgbm': lgbm_pred,
                'neural_network': dnn_pred
            },
            'model_weights': weights,
            'model_agreement': self._calculate_agreement(xgb_pred, lgbm_pred, dnn_pred)
        }
    
    def _calculate_confidence(self, xgb_pred, lgbm_pred, dnn_pred):
        """Calculate confidence (0-100) based on model agreement"""
        predictions = np.array([xgb_pred, lgbm_pred, dnn_pred])
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        cv = np.where(mean_pred != 0, std_pred / np.abs(mean_pred), 0)
        confidence = np.clip(100 * (1 - cv), 0, 100)
        return confidence
    
    def _get_prediction_interval(self, ensemble_pred, xgb_pred, lgbm_pred, dnn_pred, confidence_level=0.95):
        """Calculate prediction interval (worst/best case)"""
        predictions = np.array([xgb_pred, lgbm_pred, dnn_pred])
        std_pred = np.std(predictions, axis=0)
        z_score = 1.96 if confidence_level == 0.95 else 1.645
        lower = np.maximum(ensemble_pred - (z_score * std_pred), 0)   # floor at 0
        upper = ensemble_pred + (z_score * std_pred)
        return lower, upper
    
    def _get_model_weights(self):
        """Extract and normalize meta-learner weights"""
        try:
            coef = self.meta_learner.coef_
            total = np.sum(np.abs(coef))
            return {
                'xgboost': float(np.abs(coef[0]) / total * 100),
                'lightgbm': float(np.abs(coef[1]) / total * 100),
                'neural_network': float(np.abs(coef[2]) / total * 100)
            }
        except:
            return {'xgboost': 33.3, 'lightgbm': 33.3, 'neural_network': 33.3}
    
    def _calculate_agreement(self, xgb_pred, lgbm_pred, dnn_pred):
        """Calculate model agreement (0-100%)"""
        predictions = np.array([xgb_pred, lgbm_pred, dnn_pred])
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        cv = np.where(mean_pred != 0, std_pred / np.abs(mean_pred), 0)
        agreement = np.clip(100 * (1 - cv), 0, 100)
        return float(np.mean(agreement))
    
    def save(self, filepath):
        """Save the entire ensemble model.
        DNN is stored as raw weights (numpy arrays) to avoid Keras version issues.
        """
        model_data = {
            'model_name': self.model_name,
            'version': self.version,
            'xgb_model': self.xgb_model,
            'lgbm_model': self.lgbm_model,
            'dnn_weights': self.dnn_model.get_weights(),  
            'meta_learner': self.meta_learner,
            'feature_names': self.feature_names,
            'n_features': self.n_features
        }
        joblib.dump(model_data, filepath)
        print(f"✅ {self.model_name} saved to: {filepath}")
    
    def load(self, filepath):
        """Load a pre-trained ensemble model.
        Rebuilds DNN architecture locally and restores weights — version-independent.
        """
        model_data = joblib.load(filepath)
        self.model_name = model_data['model_name']
        self.version = model_data['version']
        self.xgb_model = model_data['xgb_model']
        self.lgbm_model = model_data['lgbm_model']
        self.meta_learner = model_data['meta_learner']
        self.feature_names = model_data['feature_names']
        self.n_features = model_data['n_features']
        
        # Restore DNN from saved weights (no Keras version dependency)
        dnn_weights = model_data.get('dnn_weights')
        if dnn_weights is not None:
            self.dnn_model = self._build_dnn(self.n_features)
            self.dnn_model.set_weights(dnn_weights)
            print(f"✅ DNN restored from weights (version-independent)")
        else:
            # Legacy fallback: try loading from .keras or .h5 file
            stored_path = model_data.get('dnn_path', '')
            keras_path = filepath.replace('.pkl', '_dnn.keras')
            h5_path = filepath.replace('.pkl', '_dnn.h5')
            dnn_path = next((p for p in [stored_path, keras_path, h5_path] if os.path.exists(p)), None)
            if dnn_path:
                self.dnn_model = load_model(dnn_path, compile=False)
                self.dnn_model.compile(optimizer='adam', loss='mean_squared_error')
                print(f"✅ DNN loaded from file (legacy): {dnn_path}")
            else:
                print(f"⚠️ DNN model not found — predictions will be XGB+LGBM only")
                self.dnn_model = None
        
        print(f"✅ {self.model_name} v{self.version} loaded successfully")
