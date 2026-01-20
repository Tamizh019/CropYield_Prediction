from price_forecast import PriceForecaster
import os

def train_price_model():
    print("="*60)
    print("💰 TRAINING MARKET PRICE MODEL (LSTM)")
    print("="*60)
    
    # Initialize forecaster
    forecaster = PriceForecaster()
    
    # Train on a representative crop (Rice)
    # The model learns "normalized" price patterns, so it can be generalized
    # reasonably well for other crops if relative trends are similar.
    print("\n🌾 Training on Rice dataset (Universal Pattern Learning)...")
    history = forecaster.train("Rice", epochs=20, save=True)
    
    if history:
        print(f"\n✅ Model trained successfully!")
        print(f"📁 Saved to: {forecaster.model_path}")
        print("\nYou can now restart the app to enable Real Market Prices.")
    else:
        print("\n❌ Training failed.")

if __name__ == "__main__":
    train_price_model()
