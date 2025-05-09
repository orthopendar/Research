""" 
Generative model training algorithm based on the CTABGANSynthesizer for final_dataset.csv,
with GPU support if available.
"""

import os
import sys
import time
import pandas as pd
import warnings
import torch

# Add the repository root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --------------------------------------------------------------------------------
# CTABGANSynthesizer & DataPrep Imports
# --------------------------------------------------------------------------------
from model.pipeline.data_preparation import DataPrep
from model.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer

warnings.filterwarnings("ignore")

class CTABGAN():
    def __init__(self,
                 raw_csv_path=r"C:\Users\ortho\OneDrive\Desktop\CTABGAN+\CTAB-GAN-Plus\Real_Datasets\final_dataset.csv",
                 test_ratio=0.20,
                 categorical_columns=[
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
                 ],
                 log_columns=[],
                 mixed_columns={},  # No mixed columns for this dataset
                 numeric_columns=[
                    'Year of diagnosis',
                    'Survival months'
                 ],
                 problem_type={None: None}):
        
        self.__name__ = 'CTABGAN'
        
        # Check for GPU availability
        use_cuda = torch.cuda.is_available()
        print("Using GPU" if use_cuda else "Using CPU")
        
        # Create the synthesizer instance.
        # CTABGANSynthesizer sets its own device internally.
        self.synthesizer = CTABGANSynthesizer()
        
        # Load the dataset
        self.raw_df = pd.read_csv(raw_csv_path)
        self.test_ratio = test_ratio
        
        # Set column definitions
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.numeric_columns = numeric_columns
        
        # For DataPrep, treat the numeric columns as both general and non-categorical
        self.general_columns = numeric_columns
        self.non_categorical_columns = numeric_columns
        
        # Assume numeric columns are integers (adjust if necessary)
        self.integer_columns = numeric_columns
        
        self.problem_type = problem_type

    def fit(self):
        start_time = time.time()
        self.data_prep = DataPrep(
            self.raw_df,
            self.categorical_columns,
            self.log_columns,
            self.mixed_columns,
            self.general_columns,
            self.non_categorical_columns,
            self.integer_columns,
            self.problem_type,
            self.test_ratio
        )
        self.synthesizer.fit(
            train_data=self.data_prep.df,
            categorical=self.data_prep.column_types["categorical"],
            mixed=self.data_prep.column_types["mixed"],
            general=self.data_prep.column_types["general"],
            non_categorical=self.data_prep.column_types["non_categorical"],
            type=self.problem_type
        )
        end_time = time.time()
        print('Finished training in', end_time - start_time, "seconds.")

    def generate_samples(self):
        sample = self.synthesizer.sample(len(self.raw_df))
        sample_df = self.data_prep.inverse_prep(sample)
        return sample_df

if __name__ == "__main__":
    # Instantiate and train the model
    model = CTABGAN()
    model.fit()
    
    # Define the output directory for synthetic data and ensure it exists
    output_dir = r"C:\Users\ortho\OneDrive\Desktop\CTABGAN+\CTAB-GAN-Plus\Fake_Datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save synthetic data to CSV
    output_file = os.path.join(output_dir, "synthetic_final_dataset.csv")
    synthetic_df = model.generate_samples()
    synthetic_df.to_csv(output_file, index=False)
    print("Synthetic dataset saved to:", output_file)
    
    # Save the trained synthesizer for future use
    model_path = os.path.join(output_dir, "ctabgan_synthesizer.pth")
    torch.save(model.synthesizer, model_path)
    print("Trained model saved to:", model_path)
