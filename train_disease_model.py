"""
🧠 CNN Model Training for Plant Disease Detection
Part of AgriVision v3.0

Trains a MobileNetV2-based CNN on the PlantVillage dataset.
"""

import os
import numpy as np
from datetime import datetime

# Check TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    print(f"✅ TensorFlow {tf.__version__} loaded")
except ImportError:
    print("❌ TensorFlow not installed. Run: pip install tensorflow")
    exit(1)

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 15  # Matching our disease database

# Paths
DATASET_PATH = "Datasets/PlantVillage"  # You need to download this
MODEL_PATH = "models/disease_model.h5"


def create_sample_structure():
    """Create sample folder structure if dataset not present"""
    print("\n📁 Creating sample dataset structure...")
    
    classes = [
        "Tomato_Early_Blight", "Tomato_Late_Blight", "Tomato_Leaf_Mold", "Tomato_Healthy",
        "Potato_Early_Blight", "Potato_Late_Blight", "Potato_Healthy",
        "Corn_Common_Rust", "Corn_Gray_Leaf_Spot", "Corn_Healthy",
        "Rice_Brown_Spot", "Rice_Leaf_Blast", "Rice_Healthy",
        "Wheat_Leaf_Rust", "Wheat_Healthy"
    ]
    
    for split in ["train", "val"]:
        for cls in classes:
            path = os.path.join(DATASET_PATH, split, cls)
            os.makedirs(path, exist_ok=True)
    
    print(f"✅ Created structure at {DATASET_PATH}")
    print("📥 Download PlantVillage dataset and organize into train/val folders")
    print("   Kaggle: https://www.kaggle.com/datasets/emmarex/plantdisease")


def build_model(num_classes):
    """Build CNN model using MobileNetV2 transfer learning"""
    print("\n🏗️ Building CNN Model...")
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Build classifier
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Model built: {model.count_params():,} parameters")
    return model


def train_model():
    """Train the disease detection model"""
    print("\n" + "=" * 60)
    print("🌿 PLANT DISEASE CNN TRAINING")
    print("=" * 60)
    
    # Check dataset
    train_path = os.path.join(DATASET_PATH, "train")
    val_path = os.path.join(DATASET_PATH, "val")
    
    if not os.path.exists(train_path):
        print(f"❌ Dataset not found at {DATASET_PATH}")
        create_sample_structure()
        print("\n⚠️ Add images to the dataset folders and run again.")
        return None
    
    # Data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.1
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load data
    print("\n📊 Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    num_classes = len(train_generator.class_indices)
    print(f"✅ Found {num_classes} classes: {list(train_generator.class_indices.keys())}")
    
    # Build model
    model = build_model(num_classes)
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy')
    ]
    
    # Train
    print(f"\n🚀 Training for {EPOCHS} epochs...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")
    
    # Print results
    final_acc = max(history.history['val_accuracy'])
    print(f"🏆 Best Validation Accuracy: {final_acc:.2%}")
    
    return history


def create_demo_model():
    """Create a lightweight demo model for testing"""
    print("\n🎯 Creating demo model for testing...")
    
    model = build_model(NUM_CLASSES)
    
    os.makedirs('models', exist_ok=True)
    model.save(MODEL_PATH)
    
    print(f"✅ Demo model saved to {MODEL_PATH}")
    print("⚠️ Note: This is untrained. Train with real data for accuracy.")


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("🌿 AGRIVISION - Disease Detection Model Trainer")
    print("=" * 60)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        create_demo_model()
    elif len(sys.argv) > 1 and sys.argv[1] == "--structure":
        create_sample_structure()
    else:
        train_model()
