"""
🧠 CNN Model Training for Plant Disease Detection (CPU Version)
Part of AgriVision v3.0

Optimized for Stability & Compatibility:
- Uses TensorFlow 2.16+ (CPU Mode)
- tf.data.Dataset Pipeline (Prefetching/Caching for speed)
- Data Augmentation Layers
"""

import os
import json
import numpy as np
from datetime import datetime

# Force Keras 3 to use TensorFlow backend (must be set before importing keras)
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Check TensorFlow & Keras
try:
    import tensorflow as tf
    import keras
    
    # Keras 3 Direct Imports
    from keras.applications import MobileNetV2
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, GlobalAveragePooling2D, RandomFlip, RandomRotation, RandomZoom, Rescaling
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    
    # Hide GPU warnings since we are forcing CPU usage
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print(f"✅ TensorFlow {tf.__version__} loaded")
    print(f"✅ Keras {keras.__version__} loaded")

except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("   Run: pip install tensorflow keras")
    exit(1)
except Exception as e:
    print(f"❌ Unexpected Error: {e}")
    exit(1)

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32  # Standard batch size for CPU
EPOCHS = 20

# Paths
DATASET_PATH = "../Datasets/PlantVillage"
MODEL_PATH = "../models/disease_model.h5"

def build_cpu_model(num_classes):
    """Build CNN model (Standard Floating Point 32-bit)"""
    print("\n🏗️ Building CNN Model...")
    
    # Data Augmentation Layers
    data_augmentation = Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.2),
        RandomZoom(0.2),
    ])

    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    
    # Build Model
    inputs = keras.Input(shape=(224, 224, 3))
    x = Rescaling(1./255)(inputs) # Normalization
    x = data_augmentation(x)
    
    # Pass through base model
    x = base_model(x, training=False)
    
    # Classifier Head
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Model built successfully")
    return model

def train_model():
    """Train the model using tf.data pipeline (Efficient for CPU too)"""
    print("\n" + "=" * 60)
    print("🌿 PLANT DISEASE CNN TRAINING (Standard Mode)")
    print("=" * 60)
    
    train_dir = os.path.join(DATASET_PATH, 'train')
    val_dir = os.path.join(DATASET_PATH, 'val')
    
    if not os.path.exists(train_dir):
        print(f"❌ Dataset not found at {train_dir}")
        print("   Please run 'organize_dataset.py' first.")
        return

    # 1. Load Data
    print("\n📊 Loading datasets...")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"✅ Found {num_classes} classes")
    
    # 2. Performance tuning (Caching helps CPU a lot too!)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    # 3. Build & Train
    model = build_cpu_model(num_classes)
    
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)
    ]
    
    print(f"\n🚀 Starting Training (Batch Size: {BATCH_SIZE})...")
    start_time = datetime.now()
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    duration = datetime.now() - start_time
    print(f"\n🏁 Training Completed in {duration}")
    
    # Save Class Mapping for App
    with open('../models/classes.json', 'w') as f:
        json.dump(class_names, f)
    print("✅ Class mapping saved to models/classes.json")
    
    return history

if __name__ == "__main__":
    train_model()
