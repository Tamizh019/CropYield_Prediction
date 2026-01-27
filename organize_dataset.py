
import os
import shutil
import random
from pathlib import Path

# Config
BASE_DIR = Path("Datasets/PlantVillage")
SOURCE_DIR = BASE_DIR / "color"
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"

# Map source folder names (from Kaggle dataset) to our target class names (from disease_detection.py)
# Only mapping the classes we support and exist in the provided dataset
FOLDER_MAPPING = {
    # Tomato
    "Tomato___Early_blight": "Tomato_Early_Blight",
    "Tomato___Late_blight": "Tomato_Late_Blight",
    "Tomato___Leaf_Mold": "Tomato_Leaf_Mold",
    "Tomato___healthy": "Tomato_Healthy",
    
    # Potato
    "Potato___Early_blight": "Potato_Early_Blight",
    "Potato___Late_blight": "Potato_Late_Blight",
    "Potato___healthy": "Potato_Healthy",
    
    # Corn
    "Corn_(maize)___Common_rust_": "Corn_Common_Rust",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Corn_Gray_Leaf_Spot",
    "Corn_(maize)___healthy": "Corn_Healthy",
    
    # Rice and Wheat are seemingly missing from the standard PlantVillage 'color' dataset subset
    # We will skip them if not found, but if you have them in a different folder, add them here.
}

def organize_dataset():
    print("🌿 Organizing PlantVillage Dataset...")
    
    if not SOURCE_DIR.exists():
        print(f"❌ Source directory not found: {SOURCE_DIR}")
        print("   Did you extract the dataset into 'Datasets/PlantVillage'?")
        return

    # Create train/val dirs
    msg_prefix = "   "
    print(f"{msg_prefix}Creating target directories...")
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    # Counters
    moved_count = 0
    skipped_count = 0
    
    for source_name, target_name in FOLDER_MAPPING.items():
        source_path = SOURCE_DIR / source_name
        
        if not source_path.exists():
            print(f"⚠️  Source folder not found: {source_name} (Expected for class {target_name})")
            skipped_count += 1
            continue
            
        print(f"📦 Processing {target_name}...")
        
        # Get all images
        images = [f for f in os.listdir(source_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        random.shuffle(images)
        
        # Split 80/20
        split_idx = int(len(images) * 0.8)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]
        
        # Target paths
        target_train_path = TRAIN_DIR / target_name
        target_val_path = VAL_DIR / target_name
        
        os.makedirs(target_train_path, exist_ok=True)
        os.makedirs(target_val_path, exist_ok=True)
        
        # Copy files
        for img in train_imgs:
            shutil.copy2(source_path / img, target_train_path / img)
            
        for img in val_imgs:
            shutil.copy2(source_path / img, target_val_path / img)
            
        moved_count += len(images)
        print(f"{msg_prefix}Moved {len(train_imgs)} to train, {len(val_imgs)} to val")

    print(f"\n✅ Organization complete!")
    print(f"   Total images processed: {moved_count}")
    print(f"   Missing classes: {skipped_count}")
    
    if moved_count == 0:
        print("\n❌ No images were moved. Please check if 'Datasets/PlantVillage/color' is empty or has different folder names.")
        # Debug list
        print("   Folders found in color/:")
        try:
            print(os.listdir(SOURCE_DIR))
        except:
            print("   (Could not list directory)")

if __name__ == "__main__":
    organize_dataset()
