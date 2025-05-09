import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import log2, sqrt
from scipy.stats import ks_2samp, chi2_contingency, spearmanr
from sklearn.metrics import mutual_info_score

# -----------------------------
# Global File Paths
# -----------------------------
DATA_PATHS = [
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
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\real_dataset.csv",
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\TVAE1.csv",
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\TVAE2.csv",
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\TVAE3.csv",
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\TVAE4.csv",
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\TVAE5.csv",
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\TVAE6.csv",
    r"C:\Users\ortho\OneDrive\Desktop\Research\Total synthetic dataset\TVAE7.csv"
]

OUTPUT_DIR = r"C:\Users\ortho\OneDrive\Desktop\Research\Analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CSV output paths
RANKING_CSV = os.path.join(OUTPUT_DIR, "ranking_table_relationships.csv")

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
# Helper Functions for Correlation Metrics
# -----------------------------
def spearman_diff(real_df, synth_df):
    """
    Compute absolute Spearman correlation difference between
    'Year of diagnosis' and 'Survival months' for real vs. synthetic.
    """
    real_vals = real_df[NUMERIC_COLS].dropna()
    synth_vals = synth_df[NUMERIC_COLS].dropna()
    if len(real_vals) < 2 or len(synth_vals) < 2:
        return np.nan
    real_corr, _ = spearmanr(real_vals["Year of diagnosis"], real_vals["Survival months"])
    synth_corr, _ = spearmanr(synth_vals["Year of diagnosis"], synth_vals["Survival months"])
    return abs(real_corr - synth_corr)

def cramers_v(confusion_matrix):
    """Compute Cramér's V for a contingency table."""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.to_numpy().sum()
    r, k = confusion_matrix.shape
    return np.sqrt((chi2 / n) / (min(r - 1, k - 1)))

def compute_cramers_v_pairs(real_df, synth_df):
    """
    Compute the average absolute difference in Cramér’s V for specified categorical pairs.
    """
    pairs = [
        ("ICD-O-3 Hist/behav", "Primary Site - labeled"),
        ("ICD-O-3 Hist/behav", "RX Summ--Surg/Rad Seq"),
        ("ICD-O-3 Hist/behav", "Reason no cancer-directed surgery"),
        ("ICD-O-3 Hist/behav", "Radiation recode"),
        ("ICD-O-3 Hist/behav", "Chemotherapy recode"),
        ("ICD-O-3 Hist/behav", "SEER cause-specific death classification"),
        ("Age recode with <1 year olds", "ICD-O-3 Hist/behav"),
        ("Race recode (White, Black, Other)", "ICD-O-3 Hist/behav"),
        ("Sex", "ICD-O-3 Hist/behav")
    ]
    diffs = []
    for colA, colB in pairs:
        if colA not in real_df.columns or colB not in real_df.columns:
            continue
        cont_real = pd.crosstab(real_df[colA], real_df[colB])
        cv_real = cramers_v(cont_real)
        cont_synth = pd.crosstab(synth_df[colA], synth_df[colB])
        cv_synth = cramers_v(cont_synth)
        diffs.append(abs(cv_real - cv_synth))
    if diffs:
        return np.mean(diffs)
    return np.nan

def discrete_mutual_info(x, y):
    """
    Compute mutual information score for two discrete arrays (categorical or discretized).
    """
    valid_mask = (~pd.isna(x)) & (~pd.isna(y))
    x_valid = x[valid_mask].astype(str)
    y_valid = y[valid_mask].astype(str)
    if len(x_valid) < 2:
        return np.nan
    return mutual_info_score(x_valid, y_valid)

def compute_mutual_info_diff(real_df, synth_df):
    """
    Compute the average absolute difference in Mutual Information (MI) for selected pairs:
      1) (ICD-O-3 Hist/behav, Primary Site - labeled)
      2) (Year of diagnosis, Survival months) after discretizing into quartiles
      3) (ICD-O-3 Hist/behav, Survival months)
    """
    pairs = [
        ("ICD-O-3 Hist/behav", "Primary Site - labeled", True),
        ("Year of diagnosis", "Survival months", False),
        ("ICD-O-3 Hist/behav", "Survival months", True)
    ]
    diffs = []
    for colA, colB, is_cat in pairs:
        if colA not in real_df.columns or colB not in real_df.columns:
            continue
        if is_cat:
            real_mi = discrete_mutual_info(real_df[colA], real_df[colB])
            synth_mi = discrete_mutual_info(synth_df[colA], synth_df[colB])
        else:
            realA = pd.qcut(real_df[colA], q=4, duplicates="drop")
            realB = pd.qcut(real_df[colB], q=4, duplicates="drop")
            synthA = pd.qcut(synth_df[colA], q=4, duplicates="drop")
            synthB = pd.qcut(synth_df[colB], q=4, duplicates="drop")
            real_mi = discrete_mutual_info(realA, realB)
            synth_mi = discrete_mutual_info(synthA, synthB)
        if pd.isna(real_mi) or pd.isna(synth_mi):
            diffs.append(np.nan)
        else:
            diffs.append(abs(real_mi - synth_mi))
    if diffs:
        return np.nanmean(diffs)
    return np.nan

# -----------------------------
# Main Execution: Compute Metrics and Create Ranking Table
# -----------------------------
def main():
    # Identify real dataset (contains "real_dataset.csv")
    real_path = [p for p in DATA_PATHS if "real_dataset.csv" in p.lower()][0]
    real_data = pd.read_csv(real_path)
    real_data.columns = real_data.columns.str.strip()
    for col in NUMERIC_COLS:
        real_data[col] = pd.to_numeric(real_data[col], errors="coerce")
    for col in CATEGORICAL_COLS:
        real_data[col] = real_data[col].astype(str)
    
    # Collect correlation results for synthetic datasets
    results = []
    synth_paths = [p for p in DATA_PATHS if "real_dataset.csv" not in p.lower()]
    for path in synth_paths:
        try:
            synth_df = pd.read_csv(path)
            synth_df.columns = synth_df.columns.str.strip()
            for col in NUMERIC_COLS:
                synth_df[col] = pd.to_numeric(synth_df[col], errors="coerce")
            for col in CATEGORICAL_COLS:
                synth_df[col] = synth_df[col].astype(str)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

        model_name = os.path.basename(path)
        sp_diff = spearman_diff(real_data, synth_df)
        cv_diff = compute_cramers_v_pairs(real_data, synth_df)
        mi_diff = compute_mutual_info_diff(real_data, synth_df)
        results.append({
            "dataset": model_name,
            "spearman_diff": sp_diff,
            "cramers_v_diff": cv_diff,
            "mi_diff": mi_diff
        })
    
    df_results = pd.DataFrame(results)
    ranking_csv = os.path.join(OUTPUT_DIR, "ranking_table_relationships.csv")
    df_results.to_csv(ranking_csv, index=False)
    print(f"Ranking table saved to: {ranking_csv}")
    
    # -----------------------------
    # Visualization: Top 10 Models per Metric
    # -----------------------------
    # Top 10 by Spearman Diff (lower is better)
    df_top_spearman = df_results.sort_values("spearman_diff", ascending=True).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_top_spearman, x="spearman_diff", y="dataset", palette="Blues_r")
    plt.xlabel("Absolute Spearman Diff (Lower is Better)")
    plt.title("Top 10 Models by Spearman Correlation Difference")
    spearman_plot_path = os.path.join(OUTPUT_DIR, "top10_spearman.png")
    plt.tight_layout()
    plt.savefig(spearman_plot_path, dpi=600)
    plt.close()
    print(f"Top 10 Spearman plot saved to: {spearman_plot_path}")

    # Top 10 by Cramér's V Diff (lower is better)
    df_top_cramers = df_results.sort_values("cramers_v_diff", ascending=True).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_top_cramers, x="cramers_v_diff", y="dataset", palette="Greens_r")
    plt.xlabel("Average Cramér's V Diff (Lower is Better)")
    plt.title("Top 10 Models by Cramér's V Difference")
    cramers_plot_path = os.path.join(OUTPUT_DIR, "top10_cramers_v.png")
    plt.tight_layout()
    plt.savefig(cramers_plot_path, dpi=600)
    plt.close()
    print(f"Top 10 Cramér's V plot saved to: {cramers_plot_path}")

    # Top 10 by Mutual Information Diff (lower is better)
    df_top_mi = df_results.sort_values("mi_diff", ascending=True).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_top_mi, x="mi_diff", y="dataset", palette="Purples_r")
    plt.xlabel("Average MI Diff (Lower is Better)")
    plt.title("Top 10 Models by Mutual Information Difference")
    mi_plot_path = os.path.join(OUTPUT_DIR, "top10_mutual_info.png")
    plt.tight_layout()
    plt.savefig(mi_plot_path, dpi=600)
    plt.close()
    print(f"Top 10 Mutual Information plot saved to: {mi_plot_path}")

if __name__ == "__main__":
    main()
