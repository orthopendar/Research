import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, chi2_contingency, spearmanr
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import MinMaxScaler
from math import log2, sqrt

# ============================================================================
# Global File Paths
# ============================================================================
DATA_PATHS = [
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\TVAE1.csv",
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\TVAE2.csv",
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\TVAE3.csv",
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\TVAE4.csv",
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\TVAE5.csv",
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\TVAE6.csv",
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\TVAE7.csv",
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\CTABGAN1.csv",
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\CTABGAN2.csv",
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\CTABGAN3.csv",
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\CTABGAN4.csv",
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\CTABGAN5.csv",
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\CTABGAN6.csv",
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\CTABGAN7.csv",
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\CTGAN1.csv",
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\CTGAN2.csv",
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\CTGAN3.csv",
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\CTGAN4.csv",
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\CTGAN5.csv",
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\CTGAN6.csv",
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\CTGAN7.csv",
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\real_dataset.csv"
]
OUTPUT_DIR = r"C:\Users\ortho\OneDrive\Desktop\Research\Analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Columns of Interest
# ============================================================================
NUMERIC_COLS = ["Year of diagnosis", "Survival months"]
CATEGORICAL_COLS = [
    "Age recode with <1 year olds",
    "Race recode (White, Black, Other)",
    "Sex",
    "ICD-O-3 Hist/behav",
    "Primary Site - labeled",
    "RX Summ--Surg/Rad Seq",
    "Reason no cancer-directed surgery",
    "Radiation recode",
    "Chemotherapy recode",
    "SEER cause-specific death classification",
    "Cause of death to site recode"
]

# ============================================================================
# Helper Functions for Correlation Metrics
# ============================================================================
def spearman_diff(real_df, synth_df):
    """Compute absolute difference in Spearman correlation between 'Year of diagnosis' and 'Survival months'."""
    real_corr, _ = spearmanr(real_df["Year of diagnosis"], real_df["Survival months"])
    synth_corr, _ = spearmanr(synth_df["Year of diagnosis"], synth_df["Survival months"])
    return abs(real_corr - synth_corr)

def cramers_v(conf_matrix):
    """Compute Cramér's V for a contingency table."""
    chi2 = chi2_contingency(conf_matrix)[0]
    n = conf_matrix.to_numpy().sum()
    r, k = conf_matrix.shape
    return np.sqrt((chi2 / n) / (min(r - 1, k - 1)))

def compute_cramers_v_pairs(real_df, synth_df):
    """
    Compute the absolute difference in Cramér’s V for the categorical pair:
    ("ICD-O-3 Hist/behav", "Primary Site - labeled")
    """
    table_real = pd.crosstab(real_df["ICD-O-3 Hist/behav"], real_df["Primary Site - labeled"])
    table_synth = pd.crosstab(synth_df["ICD-O-3 Hist/behav"], synth_df["Primary Site - labeled"])
    cv_real = cramers_v(table_real)
    cv_synth = cramers_v(table_synth)
    return abs(cv_real - cv_synth)

def discrete_mutual_info(x, y):
    """Compute mutual information for two discrete arrays."""
    valid_mask = (~pd.isna(x)) & (~pd.isna(y))
    x_valid = x[valid_mask].astype(str)
    y_valid = y[valid_mask].astype(str)
    if len(x_valid) < 2:
        return np.nan
    return mutual_info_score(x_valid, y_valid)

def compute_mutual_info_diff(real_df, synth_df):
    """
    Compute the absolute difference in Mutual Information (MI) for the categorical pair:
    ("ICD-O-3 Hist/behav", "Primary Site - labeled")
    """
    mi_real = discrete_mutual_info(real_df["ICD-O-3 Hist/behav"], real_df["Primary Site - labeled"])
    mi_synth = discrete_mutual_info(synth_df["ICD-O-3 Hist/behav"], synth_df["Primary Site - labeled"])
    return abs(mi_real - mi_synth)

# ============================================================================
# Helper Functions for Univariate Metrics
# ============================================================================
def univariate_metrics(real_df, synth_df):
    """
    Compute univariate metrics:
      - Average KS statistic for numeric columns.
      - Average Chi-square statistic for categorical columns.
    """
    metrics = {"avg_ks_stat": 0.0, "avg_chi2": 0.0, "n_numeric": 0, "n_categorical": 0}
    # KS tests for numeric columns
    for col in NUMERIC_COLS:
        real_vals = real_df[col].dropna()
        synth_vals = synth_df[col].dropna()
        if len(real_vals) > 1 and len(synth_vals) > 1:
            ks_stat, _ = ks_2samp(real_vals, synth_vals)
            metrics["avg_ks_stat"] += ks_stat
            metrics["n_numeric"] += 1
    if metrics["n_numeric"] > 0:
        metrics["avg_ks_stat"] /= metrics["n_numeric"]
    
    # Chi-square tests for categorical columns
    for col in CATEGORICAL_COLS:
        freq_real = real_df[col].value_counts().sort_index()
        freq_synth = synth_df[col].value_counts().sort_index()
        contingency = pd.DataFrame({"real": freq_real, "synth": freq_synth}).fillna(0)
        if contingency.shape[0] > 1:
            chi2, _ , _ , _ = chi2_contingency(contingency)
            metrics["avg_chi2"] += chi2
            metrics["n_categorical"] += 1
    if metrics["n_categorical"] > 0:
        metrics["avg_chi2"] /= metrics["n_categorical"]
    return metrics

# ============================================================================
# Main Composite Analysis: Compute Metrics and Generate Composite Rankings
# ============================================================================
def main():
    # Load the real dataset
    real_path = [p for p in DATA_PATHS if "real_dataset.csv" in p.lower()][0]
    real_df = pd.read_csv(real_path)
    real_df.columns = real_df.columns.str.strip()
    for col in NUMERIC_COLS:
        real_df[col] = pd.to_numeric(real_df[col], errors='coerce')
    for col in CATEGORICAL_COLS:
        real_df[col] = real_df[col].astype(str)
    
    # Process synthetic datasets and compute metrics
    synth_paths = [p for p in DATA_PATHS if "real_dataset.csv" not in p.lower()]
    composite_results = []
    
    for path in synth_paths:
        dataset_name = os.path.basename(path)
        try:
            synth_df = pd.read_csv(path)
            synth_df.columns = synth_df.columns.str.strip()
            for col in NUMERIC_COLS:
                synth_df[col] = pd.to_numeric(synth_df[col], errors='coerce')
            for col in CATEGORICAL_COLS:
                synth_df[col] = synth_df[col].astype(str)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

        # Compute univariate metrics
        uni = univariate_metrics(real_df, synth_df)
        # Compute correlation metrics
        sp_diff = spearman_diff(real_df, synth_df)
        cv_diff = compute_cramers_v_pairs(real_df, synth_df)
        mi_diff = compute_mutual_info_diff(real_df, synth_df)
        
        composite_results.append({
            "dataset": dataset_name,
            "avg_ks_stat": uni["avg_ks_stat"],
            "avg_chi2": uni["avg_chi2"],
            "spearman_diff": sp_diff,
            "cramers_v_diff": cv_diff,
            "mi_diff": mi_diff
        })
    
    df_comp = pd.DataFrame(composite_results)
    
    # Normalize and invert the metrics (lower differences are better, so we invert after scaling)
    scaler = MinMaxScaler()
    metrics_to_norm = ["avg_ks_stat", "avg_chi2", "spearman_diff", "cramers_v_diff", "mi_diff"]
    df_norm = pd.DataFrame(scaler.fit_transform(df_comp[metrics_to_norm]), columns=metrics_to_norm)
    for col in metrics_to_norm:
        df_norm[col] = 1 - df_norm[col]
    for col in metrics_to_norm:
        df_comp[col + "_score"] = df_norm[col]
    
    score_cols = [col + "_score" for col in metrics_to_norm]
    df_comp["composite_score"] = df_comp[score_cols].mean(axis=1)
    
    # Sort datasets by composite score (higher is better)
    df_ranked = df_comp.sort_values("composite_score", ascending=False)
    
    
    # ---------------------------
    # Visualization: Composite Bar Plot for Top Ten Models
    # ---------------------------
    top10 = df_ranked.head(10)
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top10, x="composite_score", y="dataset", palette="Blues_r")
    plt.xlabel("Composite Score (Higher = Better)")
    plt.ylabel("Synthetic Dataset")
    plt.title("Composite Model Rankings (Top 10)")
    plt.tight_layout()
    composite_plot_path = os.path.join(OUTPUT_DIR, "composite_model_ranking_top10.png")
    plt.savefig(composite_plot_path, dpi=300)
    plt.close()
    print("Composite ranking plot for top 10 saved to:", composite_plot_path)
    

if __name__ == "__main__":
    main()
