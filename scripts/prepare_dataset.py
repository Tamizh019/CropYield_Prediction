import pandas as pd
import sys

def prepare_dataset():
    """
    Intelligent dataset preparation 
    Specifically handles user's dataset format with specialized column names
    """
    print("=" * 60)
    print("YieldMax Dataset Preparation")
    print("=" * 60)
    
    # Load dataset
    input_path = 'Datasets/Crop_yield_india.csv'
    output_path = 'Datasets/Yield_Data_Ready.csv'
    
    try:
        df = pd.read_csv(input_path)
        print(f"✓ Loaded: {len(df)} records from {input_path}")
    except FileNotFoundError:
        print(f"✗ Error: {input_path} not found!")
        sys.exit(1)
    
    print("\nProcessing columns...")
    
    # 1. Standardize column names (strip spaces first)
    df.columns = df.columns.str.strip()
    
    # 2. Rename based on user's exact file structure
    rename_map = {
        'Dist Name': 'District_Name',
        'State Name': 'State_Name',
        'Year': 'Crop_Year',
        'Area_ha': 'Area',
        'Yield_kg_per_ha': 'Yield_Rate',  # Temporary name
        'Temperature_C': 'Temperature',
        'Humidity_%': 'Humidity',
        'Rainfall_mm': 'Rainfall'
    }
    
    # Check if required source columns exist
    missing_source = [col for col in rename_map.keys() if col not in df.columns]
    
    # Allow for some flexibility if already renamed or standard
    final_map = {}
    for source, target in rename_map.items():
        if source in df.columns:
            final_map[source] = target
        elif target in df.columns:
            print(f"  - {target} already exists")
    
    if final_map:
        df.rename(columns=final_map, inplace=True)
        print(f"✓ Renamed {len(final_map)} columns to standard format")
        
    # 3. Calculate Production (Yield * Area)
    # We need Production as the target variable (in Tonnes)
    if 'Production' not in df.columns:
        if 'Yield_Rate' in df.columns and 'Area' in df.columns:
            # Yield is kg/ha, Area is ha. Production = (kg/ha * ha) / 1000 = Tonnes
            df['Production'] = (df['Yield_Rate'] * df['Area']) / 1000
            print("✓ Calculated 'Production' (Tonnes) from Yield * Area")
        else:
            print("✗ Error: Cannot calculate Production! Missing 'Yield_kg_per_ha' or 'Area_ha'")
            print(f"  Available columns: {list(df.columns)}")
            sys.exit(1)
            
    # 4. Handle missing values
    initial_nulls = df.isnull().sum().sum()
    if initial_nulls > 0:
        print(f"✓ Found {initial_nulls} missing values, handling them...")
        for col in ['Temperature', 'Humidity', 'pH', 'Rainfall']:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
    
    # 5. Filter valid rows
    required_cols = [
        'State_Name', 'District_Name', 'Crop_Year', 'Crop',
        'Area', 'Production', 'Temperature', 'Humidity', 'pH', 'Rainfall'
    ]
    
    # Check if we have all required columns now
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        print(f"✗ Error: Missing final columns: {missing_required}")
        sys.exit(1)
        
    df = df.dropna(subset=required_cols)
    df = df[df['Production'] < 500000] # Remove outliers
    df = df[df['Area'] > 0]
    
    # 6. Select and Save
    df_final = df[required_cols].copy()
    df_final.to_csv(output_path, index=False)
    
    print(f"\n{'=' * 60}")
    print(f"✓ SUCCESS: Dataset saved to {output_path}")
    print(f"✓ Records: {len(df_final)}")
    print(f"✓ Columns: {list(df_final.columns)}")
    print(f"{'=' * 60}")
    print("\nNext: Run python scripts/train_ensemble.py")

if __name__ == '__main__':
    prepare_dataset()
