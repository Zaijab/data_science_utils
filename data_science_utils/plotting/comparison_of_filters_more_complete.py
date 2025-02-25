import pandas as pd

enkf_df = pd.read_csv('cache/enkf_1_4_cov.csv')
gmf_df = pd.read_csv('cache/initial_cov_1_4.csv')

print(enkf_df)
print(gmf_df)

import pandas as pd
import numpy as np
import ast
from fractions import Fraction
import matplotlib.pyplot as plt
import seaborn as sns

def format_fraction(x):
    frac = Fraction(x).limit_denominator(1000)
    return f"{frac.numerator}" if frac.denominator == 1 else f"{frac.numerator}/{frac.denominator}"

# -------------------------------
# Load and process the CSV files
# -------------------------------

# Load enkf CSV.
enkf_df = pd.read_csv('cache/enkf_1_4_cov.csv')
# Rename columns: measurement_covariance -> covariance, update_method -> method.
enkf_df = enkf_df.rename(columns={
    'measurement_covariance': 'covariance',
    'update_method': 'method'
})
# For EnKF/BRUEnKF, bandwidth is constant (set to 1.0).
if 'bandwidth' not in enkf_df.columns:
    enkf_df['bandwidth'] = 1.0
# Add a dummy "random" column if needed.
if 'random' not in enkf_df.columns:
    enkf_df['random'] = enkf_df.index.astype(str)

# Convert the rmse column (a string like "[0.59, 2.91, ...]") into a scalar (the mean).
def process_rmse(rmse_str):
    try:
        rmse_list = ast.literal_eval(rmse_str)
        return np.mean(rmse_list)
    except Exception:
        return np.nan

enkf_df['RMSE'] = enkf_df['rmse'].apply(process_rmse)
enkf_df.drop(columns=['rmse'], inplace=True)
enkf_df['covariance'] = enkf_df['covariance'].astype(float)

# Load gmf CSV.
gmf_df = pd.read_csv('cache/initial_cov_1_4.csv')
# Rename "sampling" to "method" (gmf_df uses different names).
gmf_df = gmf_df.rename(columns={'sampling': 'method'})
gmf_df['covariance'] = gmf_df['covariance'].astype(float)
gmf_df['bandwidth'] = gmf_df['bandwidth'].astype(float)
# Assume gmf_df already has a scalar RMSE column.

# -------------------------------
# Combine dataframes and map method names
# -------------------------------

df_all = pd.concat([enkf_df, gmf_df], ignore_index=True)

# Map raw method names to the four desired labels.
# For enkf_df, method values are "enkf_update" or "bruenkf_update".
# For gmf_df, method values are "ikeda_rejection_sample_batch" or "sample_gaussian_mixture".
method_map = {
    "enkf_update": "EnKF",
    "bruenkf_update": "BRUEnKF",
    "ikeda_rejection_sample_batch": "DI-EnGMF",
    "sample_gaussian_mixture": "EnGMF"
}
df_all['sampling_label'] = df_all['method'].map(method_map)

# -------------------------------
# Aggregate RMSE over Monte Carlo runs
# -------------------------------

# Group by bandwidth, covariance, ensemble_size, and sampling_label.
grouped = df_all.groupby(["bandwidth", "covariance", "ensemble_size", "sampling_label"])["RMSE"]
df_stats = grouped.agg(mean_rmse=("mean"), std_rmse=("std")).reset_index()

# Create fraction labels for bandwidth and covariance (for facet labels).
df_stats["bandwidth_str"] = df_stats["bandwidth"].apply(format_fraction)
df_stats["covariance_str"] = df_stats["covariance"].apply(format_fraction)

# -------------------------------
# Plot the facet grid
# -------------------------------

sns.set(context="talk", style="whitegrid", font_scale=1.2)
palette = sns.color_palette("colorblind", n_colors=df_stats["sampling_label"].nunique())

g = sns.FacetGrid(
    data=df_stats,
    row="covariance_str",    # 3 unique bandwidth values → 3 rows
    hue="sampling_label",   # 4 methods → 4 lines per subplot
    margin_titles=True,
    sharey=False,
    palette=palette,
    height=4,
    aspect=1.3
)

g.map(
    plt.errorbar,
    "ensemble_size", "mean_rmse", "std_rmse",
    marker="o", capsize=3, linestyle="--"
)

g.add_legend(title="Method")
g.fig.subplots_adjust(right=0.8)
if g._legend:
    g._legend.set_bbox_to_anchor((0.98, 0.5))
    g._legend.set_loc("center left")
g.set_axis_labels(r"\textbf{Ensemble Size}", r"\textbf{RMSE}")
g.set_titles(
    row_template=r"Bandwidth ($\beta$): {row_name}",
    col_template=r"Covariance ($R$): {col_name}"
)
g._legend.set_in_layout(True)
plt.tight_layout()
g.fig.savefig("plots/all_methods_rmse_plot.svg", format="svg", bbox_inches="tight")
plt.show()
