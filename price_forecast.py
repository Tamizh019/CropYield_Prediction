"""
💰 Market Price Forecasting Module
Part of AgriVision v3.0

Uses LSTM (Long Short-Term Memory) neural network to predict crop prices.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not installed. Price forecast will use simulation mode.")


# ========================================
# HISTORICAL PRICE DATA (Sample)
# ========================================

# Sample price data for Indian crops (INR per quintal)
CROP_BASE_PRICES = {
    "Rice": {"base": 2200, "volatility": 0.08, "trend": 0.03},
    "Wheat": {"base": 2400, "volatility": 0.06, "trend": 0.025},
    "Maize": {"base": 1900, "volatility": 0.10, "trend": 0.02},
    "Cotton": {"base": 6500, "volatility": 0.12, "trend": 0.04},
    "Sugarcane": {"base": 350, "volatility": 0.05, "trend": 0.02},
    "Soybean": {"base": 4500, "volatility": 0.15, "trend": 0.03},
    "Potato": {"base": 1200, "volatility": 0.20, "trend": 0.01},
    "Onion": {"base": 1500, "volatility": 0.25, "trend": 0.02},
    "Tomato": {"base": 2000, "volatility": 0.30, "trend": 0.01},
    "Groundnut": {"base": 5800, "volatility": 0.10, "trend": 0.035},
    "Mustard": {"base": 5200, "volatility": 0.08, "trend": 0.03},
    "Chickpea": {"base": 5500, "volatility": 0.09, "trend": 0.025},
    "Banana": {"base": 1800, "volatility": 0.15, "trend": 0.02},
    "Mango": {"base": 4000, "volatility": 0.18, "trend": 0.03},
}

# Seasonal factors (month -> multiplier)
SEASONAL_FACTORS = {
    1: 1.05,   # January - Post harvest, slight premium
    2: 1.02,   # February
    3: 0.98,   # March - New harvest arriving
    4: 0.95,   # April - Peak supply
    5: 0.97,   # May
    6: 1.00,   # June
    7: 1.03,   # July - Monsoon uncertainty
    8: 1.05,   # August
    9: 1.02,   # September
    10: 0.98,  # October - Kharif harvest
    11: 0.96,  # November - Peak supply
    12: 1.00,  # December
}


# ========================================
# PRICE GENERATOR (For Training Data)
# ========================================

def generate_historical_prices(crop, days=365):
    """
    Generate realistic historical price data for a crop
    
    Args:
        crop: Crop name
        days: Number of days of history
    
    Returns:
        DataFrame with Date and Price columns
    """
    if crop not in CROP_BASE_PRICES:
        crop = "Rice"  # Default
    
    params = CROP_BASE_PRICES[crop]
    base_price = params["base"]
    volatility = params["volatility"]
    trend = params["trend"]
    
    # Generate dates
    end_date = datetime.now()
    dates = [end_date - timedelta(days=x) for x in range(days)][::-1]
    
    # Generate prices with trend, seasonality, and randomness
    prices = []
    current_price = base_price * 0.9  # Start lower than current
    
    for i, date in enumerate(dates):
        # Trend component
        trend_factor = 1 + (trend * i / 365)
        
        # Seasonal component
        seasonal = SEASONAL_FACTORS.get(date.month, 1.0)
        
        # Random walk component
        random_change = np.random.normal(0, volatility * base_price / 30)
        
        # Calculate price
        price = current_price * trend_factor * seasonal + random_change
        price = max(price, base_price * 0.5)  # Floor
        price = min(price, base_price * 2.0)  # Ceiling
        
        prices.append(round(price, 2))
        current_price = price / (trend_factor * seasonal)  # Detrend for next iteration
    
    return pd.DataFrame({
        "Date": dates,
        "Price": prices,
        "Crop": crop
    })


# ========================================
# LSTM PRICE FORECASTER
# ========================================

class PriceForecaster:
    """
    LSTM-based crop price forecasting
    """
    
    def __init__(self, model_path='models/price_lstm.h5'):
        self.model_path = model_path
        self.model = None
        self.scaler = MinMaxScaler() if TF_AVAILABLE else None
        self.sequence_length = 30  # Use 30 days to predict next 7
        self.forecast_days = 7
        
        # Load model if available
        self._load_model()
    
    def _load_model(self):
        """Load trained LSTM model"""
        if not TF_AVAILABLE:
            print("⚠️ TensorFlow not available - using simulation mode")
            return
        
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                print(f"✅ Price forecasting model loaded")
            except Exception as e:
                print(f"❌ Failed to load price model: {e}")
        else:
            print(f"⚠️ Price model not found. Using simulation mode.")
    
    def create_sequences(self, data):
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length - self.forecast_days + 1):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length:i + self.sequence_length + self.forecast_days])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build LSTM model architecture"""
        if not TF_AVAILABLE:
            return None
        
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(self.forecast_days)  # Predict 7 days
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, crop, epochs=50, save=True):
        """
        Train LSTM model on historical data
        
        Args:
            crop: Crop name
            epochs: Training epochs
            save: Whether to save the model
        """
        if not TF_AVAILABLE:
            print("❌ TensorFlow required for training")
            return None
        
        print(f"\n📈 Training price forecaster for {crop}...")
        
        # Generate training data
        df = generate_historical_prices(crop, days=730)  # 2 years
        prices = df['Price'].values.reshape(-1, 1)
        
        # Scale data
        scaled_prices = self.scaler.fit_transform(prices)
        
        # Create sequences
        X, y = self.create_sequences(scaled_prices.flatten())
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Split train/test
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Build and train model
        self.model = self.build_model((self.sequence_length, 1))
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Evaluate
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"✅ Training complete! Test MAE: {mae:.4f}")
        
        # Save model
        if save:
            os.makedirs('models', exist_ok=True)
            self.model.save(self.model_path)
            print(f"💾 Model saved to {self.model_path}")
        
        return history
    
    def forecast(self, crop, include_history=True):
        """
        Forecast prices for the next 7 days
        
        Args:
            crop: Crop name
            include_history: Include last 30 days in response
        
        Returns:
            dict with forecast results
        """
        try:
            # Get historical data
            df = generate_historical_prices(crop, days=60)
            recent_prices = df['Price'].values[-self.sequence_length:]
            
            if self.model is not None and TF_AVAILABLE:
                # Real LSTM prediction
                scaled = self.scaler.fit_transform(recent_prices.reshape(-1, 1))
                X = scaled.reshape(1, self.sequence_length, 1)
                
                prediction_scaled = self.model.predict(X, verbose=0)
                forecast_prices = self.scaler.inverse_transform(
                    prediction_scaled.reshape(-1, 1)
                ).flatten()
            else:
                # Simulation mode - intelligent forecast
                last_price = recent_prices[-1]
                params = CROP_BASE_PRICES.get(crop, CROP_BASE_PRICES["Rice"])
                
                forecast_prices = []
                current = last_price
                
                for i in range(self.forecast_days):
                    # Add trend and small random variation
                    change = np.random.normal(0.002, 0.01) * current
                    current = current + change
                    forecast_prices.append(round(current, 2))
            
            # Generate forecast dates
            last_date = df['Date'].iloc[-1]
            forecast_dates = [
                (last_date + timedelta(days=i+1)).strftime('%Y-%m-%d')
                for i in range(self.forecast_days)
            ]
            
            # Calculate metrics
            avg_forecast = np.mean(forecast_prices)
            current_price = recent_prices[-1]
            price_change = ((avg_forecast - current_price) / current_price) * 100
            
            # Recommendation
            if price_change > 5:
                recommendation = "HOLD - Prices expected to rise. Wait before selling."
                rec_type = "bullish"
            elif price_change < -5:
                recommendation = "SELL NOW - Prices may decline. Consider selling soon."
                rec_type = "bearish"
            else:
                recommendation = "NEUTRAL - Stable prices expected. Sell as per convenience."
                rec_type = "neutral"
            
            result = {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "crop": crop,
                "current_price": round(current_price, 2),
                "forecast": {
                    "dates": forecast_dates,
                    "prices": [round(p, 2) for p in forecast_prices],
                    "average": round(avg_forecast, 2),
                    "change_percent": round(price_change, 2)
                },
                "recommendation": {
                    "text": recommendation,
                    "type": rec_type
                },
                "history": {
                    "dates": [d.strftime('%Y-%m-%d') for d in df['Date'].tail(30)],
                    "prices": recent_prices.tolist()
                } if include_history else None,
                "model_used": "LSTM Neural Network" if self.model else "Statistical Simulation"
            }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_market_overview(self):
        """
        Get price overview for all supported crops
        """
        overview = []
        
        for crop in CROP_BASE_PRICES.keys():
            df = generate_historical_prices(crop, days=30)
            prices = df['Price'].values
            
            current = prices[-1]
            week_ago = prices[-7] if len(prices) >= 7 else prices[0]
            month_ago = prices[0]
            
            week_change = ((current - week_ago) / week_ago) * 100
            month_change = ((current - month_ago) / month_ago) * 100
            
            overview.append({
                "crop": crop,
                "current_price": round(current, 2),
                "week_change": round(week_change, 2),
                "month_change": round(month_change, 2),
                "trend": "up" if week_change > 0 else "down" if week_change < 0 else "stable"
            })
        
        # Sort by week change
        overview.sort(key=lambda x: x['week_change'], reverse=True)
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "market_data": overview,
            "top_gainer": overview[0]['crop'] if overview else None,
            "top_loser": overview[-1]['crop'] if overview else None
        }
    
    def get_supported_crops(self):
        """Get list of supported crops"""
        return list(CROP_BASE_PRICES.keys())


# ========================================
# SINGLETON INSTANCE
# ========================================

price_forecaster = None

def get_price_forecaster():
    """Get or create PriceForecaster instance"""
    global price_forecaster
    if price_forecaster is None:
        price_forecaster = PriceForecaster()
    return price_forecaster


# ========================================
# CLI TEST
# ========================================

if __name__ == "__main__":
    print("=" * 50)
    print("💰 MARKET PRICE FORECASTER - Test Mode")
    print("=" * 50)
    
    forecaster = get_price_forecaster()
    
    print(f"\n📋 Supported Crops: {forecaster.get_supported_crops()}")
    
    # Test forecast
    print("\n🧪 Testing price forecast for Rice...")
    result = forecaster.forecast("Rice")
    
    if result['success']:
        print(f"\n📊 Forecast Result:")
        print(f"   Current Price: ₹{result['current_price']}/quintal")
        print(f"   7-Day Avg Forecast: ₹{result['forecast']['average']}/quintal")
        print(f"   Change: {result['forecast']['change_percent']:+.2f}%")
        print(f"\n💡 Recommendation: {result['recommendation']['text']}")
    
    # Test market overview
    print("\n📈 Market Overview...")
    overview = forecaster.get_market_overview()
    print(f"   Top Gainer: {overview['top_gainer']}")
    print(f"   Top Loser: {overview['top_loser']}")
