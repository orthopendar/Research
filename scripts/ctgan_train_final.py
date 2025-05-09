import os
import sys
import pandas as pd

# --------------------------------------------------------------------------------
# 0. Update sys.path to include the parent directory of the 'ctgan' package
# --------------------------------------------------------------------------------
# Assuming your file is located at:
# C:\Users\ortho\OneDrive\Desktop\SDV\CTGAN\my_project\scripts\ctgan_train_final.py
# and your 'ctgan' package is at:
# C:\Users\ortho\OneDrive\Desktop\SDV\CTGAN\ctgan
# We need to add: C:\Users\ortho\OneDrive\Desktop\SDV\CTGAN to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --------------------------------------------------------------------------------
# SDV Imports
# --------------------------------------------------------------------------------
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

# --------------------------------------------------------------------------------
# 1. Define File Paths
# --------------------------------------------------------------------------------
DATASET_PATH = r"C:\Users\ortho\OneDrive\Desktop\SDV\CTGAN\my_project\data\final_dataset.csv"
OUTPUT_DIR = r"C:\Users\ortho\OneDrive\Desktop\SDV\CTGAN\my_project\output"
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.json")
SYNTHETIC_DATA_PATH = os.path.join(OUTPUT_DIR, "synthetic_data.csv")
MODEL_PATH = os.path.join(OUTPUT_DIR, "my_synthesizer.pkl")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------------
# 2. Load the Dataset
# --------------------------------------------------------------------------------
df = pd.read_csv(DATASET_PATH)
print("✅ Dataset loaded successfully!")
print(df.head())
print(f"Dataset shape: {df.shape}")

# --------------------------------------------------------------------------------
# 3. Initialize Metadata and Auto-Detect
# --------------------------------------------------------------------------------
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)
print("✅ Metadata auto‐detected.")

# --------------------------------------------------------------------------------
# 4. Update Column Types
# --------------------------------------------------------------------------------
# Updated Categorical Columns
categorical_columns = [
    'Age recode with <1 year olds',
    'Race recode (White, Black, Other)',
    'Sex',
    'ICD-O-3 Hist/behav',
    'Primary Site - labeled',
    'RX Summ--Surg/Rad Seq',
    'Reason no cancer-directed surgery',
    'Radiation recode',
    'Chemotherapy recode',
    'SEER cause-specific death classification',
    'Cause of death to site recode'
]

# Updated Numeric Columns
numeric_columns = [
    'Year of diagnosis',
    'Survival months'
]

# Apply Categorical Types
for col in categorical_columns:
    metadata.update_column(column_name=col, sdtype='categorical')

# Apply Numeric Types
for col in numeric_columns:
    metadata.update_column(column_name=col, sdtype='numerical')

print("✅ Column types updated successfully!")

# --------------------------------------------------------------------------------
# 5. Validate and Save Metadata
# --------------------------------------------------------------------------------
try:
    metadata.validate()
    print("✅ Metadata validation successful!")
except Exception as e:
    print("❌ Metadata validation failed:", e)

# Remove existing metadata.json file if it exists
if os.path.exists(METADATA_PATH):
    os.remove(METADATA_PATH)
    print(f"🗑 Existing metadata file removed: {METADATA_PATH}")

metadata.save_to_json(filepath=METADATA_PATH)
print(f"📁 Metadata saved to: {METADATA_PATH}")

# --------------------------------------------------------------------------------
# 6. Load Metadata from JSON (Optional, but recommended for a clean workflow)
# --------------------------------------------------------------------------------
metadata = SingleTableMetadata.load_from_json(METADATA_PATH)
print("✅ Metadata reloaded from JSON.")

# --------------------------------------------------------------------------------
# 7. Train the CTGAN Synthesizer
# --------------------------------------------------------------------------------
ctgan = CTGANSynthesizer(metadata, verbose=True, epochs=3000)
ctgan.fit(df)
print("🚀 CTGAN model trained successfully!")

# Save the trained model
ctgan.save(filepath=MODEL_PATH)
print(f"🚀 CTGAN model saved successfully to {MODEL_PATH}!")

# --------------------------------------------------------------------------------
# 8. Generate and Save Synthetic Data
# --------------------------------------------------------------------------------
synthetic_data = ctgan.sample(num_rows=len(df))
print("✅ Synthetic data generated!")
print(synthetic_data.head())

synthetic_data.to_csv(SYNTHETIC_DATA_PATH, index=False)
print(f"📁 Synthetic data saved to: {SYNTHETIC_DATA_PATH}")
