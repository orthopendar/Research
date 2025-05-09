import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, chi2_contingency
from math import sqrt, log2

# -----------------------------
# Global Setup
# -----------------------------
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

# CSV output paths
RAW_CSV_PATH = os.path.join(OUTPUT_DIR, "analysis_results_raw.csv")
SORTED_KS_STAT_CSV = os.path.join(OUTPUT_DIR, "analysis_sorted_by_ks_stat.csv")
SORTED_KS_P_CSV = os.path.join(OUTPUT_DIR, "analysis_sorted_by_ks_p.csv")
SORTED_CHI2_CSV = os.path.join(OUTPUT_DIR, "analysis_sorted_by_chi2.csv")
SORTED_CHI2_P_CSV = os.path.join(OUTPUT_DIR, "analysis_sorted_by_chi2_p.csv")
COMPOSITE_CSV = os.path.join(OUTPUT_DIR, "composite_ranking.csv")

# -----------------------------
# Columns of Interest
# -----------------------------
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

# -----------------------------
# Load Real Dataset
# -----------------------------
real_path = [p for p in DATA_PATHS if "real_dataset.csv" in p.lower()][0]
real_data = pd.read_csv(real_path)
real_data.columns = real_data.columns.str.strip()
for col in NUMERIC_COLS:
    real_data[col] = pd.to_numeric(real_data[col], errors="coerce")
for col in CATEGORICAL_COLS:
    real_data[col] = real_data[col].astype(str)

# Create list of synthetic dataset paths (exclude real dataset)
synthetic_paths = [p for p in DATA_PATHS if p != real_path]

# -----------------------------
# Analysis Function: Univariate Tests
# -----------------------------
def analyze_dataset(real_df, synth_df):
    """
    Perform univariate analysis:
      - KS tests for numeric columns,
      - Chi-square tests for categorical columns.
    Returns a dictionary with average KS statistics/p-values and
    average Chi-square statistics/p-values.
    """
    metrics = {
        "avg_ks_stat": 0.0,
        "avg_ks_p": 0.0,
        "avg_chi2": 0.0,
        "avg_chi2_p": 0.0,
        "n_numeric_tests": 0,
        "n_categorical_tests": 0
    }
    for col in NUMERIC_COLS:
        real_vals = real_df[col].dropna()
        synth_vals = synth_df[col].dropna()
        if len(real_vals) > 1 and len(synth_vals) > 1:
            ks_stat, ks_p = ks_2samp(real_vals, synth_vals)
            metrics["avg_ks_stat"] += ks_stat
            metrics["avg_ks_p"] += ks_p
            metrics["n_numeric_tests"] += 1
    for col in CATEGORICAL_COLS:
        freq_real = real_df[col].value_counts().sort_index()
        freq_synth = synth_df[col].value_counts().sort_index()
        contingency = pd.DataFrame({"real": freq_real, "synth": freq_synth}).fillna(0)
        if contingency.shape[0] > 1:
            chi2, p_val, dof, _ = chi2_contingency(contingency)
            metrics["avg_chi2"] += chi2
            metrics["avg_chi2_p"] += p_val
            metrics["n_categorical_tests"] += 1
    if metrics["n_numeric_tests"] > 0:
        metrics["avg_ks_stat"] /= metrics["n_numeric_tests"]
        metrics["avg_ks_p"] /= metrics["n_numeric_tests"]
    if metrics["n_categorical_tests"] > 0:
        metrics["avg_chi2"] /= metrics["n_categorical_tests"]
        metrics["avg_chi2_p"] /= metrics["n_categorical_tests"]
    return metrics

# -----------------------------
# Main Loop: Evaluate All Synthetic Datasets
# -----------------------------
results = []
for synth_path in synthetic_paths:
    dataset_name = os.path.basename(synth_path)
    try:
        synth_data = pd.read_csv(synth_path)
        synth_data.columns = synth_data.columns.str.strip()
    except Exception as e:
        print(f"Error loading {synth_path}: {e}")
        continue
    # Convert columns for synth data
    for col in NUMERIC_COLS:
        synth_data[col] = pd.to_numeric(synth_data[col], errors="coerce")
    for col in CATEGORICAL_COLS:
        synth_data[col] = synth_data[col].astype(str)
    metrics = analyze_dataset(real_data, synth_data)
    metrics["dataset"] = dataset_name
    results.append(metrics)

df_results = pd.DataFrame(results)
df_results.to_csv(RAW_CSV_PATH, index=False)
print("Raw metrics saved to:", RAW_CSV_PATH)

# -----------------------------
# Create Sorted CSVs for Univariate Tests
# -----------------------------
df_sorted_ks_stat = df_results.sort_values("avg_ks_stat", ascending=True)
df_sorted_ks_stat.to_csv(SORTED_KS_STAT_CSV, index=False)
print("Sorted by KS statistic saved to:", SORTED_KS_STAT_CSV)

df_sorted_ks_p = df_results.sort_values("avg_ks_p", ascending=False)
df_sorted_ks_p.to_csv(SORTED_KS_P_CSV, index=False)
print("Sorted by KS p-value saved to:", SORTED_KS_P_CSV)

df_sorted_chi2 = df_results.sort_values("avg_chi2", ascending=True)
df_sorted_chi2.to_csv(SORTED_CHI2_CSV, index=False)
print("Sorted by Chi-square saved to:", SORTED_CHI2_CSV)

df_sorted_chi2_p = df_results.sort_values("avg_chi2_p", ascending=False)
df_sorted_chi2_p.to_csv(SORTED_CHI2_P_CSV, index=False)
print("Sorted by Chi-square p-value saved to:", SORTED_CHI2_P_CSV)

# -----------------------------
# Composite Ranking: Create composite score using normalized KS and Chi-square scores
# -----------------------------
def minmax_scale(series, invert=False):
    s_min, s_max = series.min(), series.max()
    if s_min == s_max:
        return pd.Series([1.0 if invert else 0.0] * len(series), index=series.index)
    scaled = (series - s_min) / (s_max - s_min)
    if invert:
        return 1 - scaled
    return scaled

df_results["ks_score"] = minmax_scale(df_results["avg_ks_stat"], invert=True)
df_results["chi2_score"] = minmax_scale(df_results["avg_chi2"], invert=True)
df_results["composite_score"] = (df_results["ks_score"] + df_results["chi2_score"]) / 2
df_composite_sorted = df_results.sort_values("composite_score", ascending=False)
df_composite_sorted.to_csv(COMPOSITE_CSV, index=False)
print("Composite ranking saved to:", COMPOSITE_CSV)

# -----------------------------
# Visualization: Top 10 Models for Univariate Tests
# -----------------------------
# Plot for KS statistic (lower is better)
top10_ks = df_sorted_ks_stat.head(10)
plt.figure(figsize=(10, 6))
sns.barplot(data=top10_ks, x="avg_ks_stat", y="dataset", palette="Blues_r")
plt.xlabel("Average KS Statistic (Lower is Better)")
plt.title("Top 10 Models by KS Statistic")
ks_plot_path = os.path.join(OUTPUT_DIR, "top10_KS.png")
plt.tight_layout()
plt.savefig(ks_plot_path, dpi=600)
plt.close()
print("Top 10 KS statistic visualization saved to:", ks_plot_path)

# Plot for Chi-square statistic (lower is better)
top10_chi2 = df_sorted_chi2.head(10)
plt.figure(figsize=(10, 6))
sns.barplot(data=top10_chi2, x="avg_chi2", y="dataset", palette="Greens_r")
plt.xlabel("Average Chi-square Statistic (Lower is Better)")
plt.title("Top 10 Models by Chi-square Statistic")
chi2_plot_path = os.path.join(OUTPUT_DIR, "top10_ChiSquare.png")
plt.tight_layout()
plt.savefig(chi2_plot_path, dpi=600)
plt.close()
print("Top 10 Chi-square visualization saved to:", chi2_plot_path)

# Additionally, visualize composite ranking for top 10 models
top10_composite = df_composite_sorted.head(10)
plt.figure(figsize=(10, 6))
sns.barplot(data=top10_composite, x="composite_score", y="dataset", palette="Purples_r")
plt.xlabel("Composite Score (Higher is Better)")
plt.title("Top 10 Models by Composite Univariate Score")
composite_plot_path = os.path.join(OUTPUT_DIR, "top10_Composite.png")
plt.tight_layout()
plt.savefig(composite_plot_path, dpi=600)
plt.close()
print("Top 10 Composite visualization saved to:", composite_plot_path)

print("All univariate analyses, composite ranking, and visualizations are complete.")
