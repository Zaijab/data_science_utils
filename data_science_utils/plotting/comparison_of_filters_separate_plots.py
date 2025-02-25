import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import os


# -------------------------------
# Utility: Format covariance label for display.
def format_cov_label(x):
    # Do not change the numeric value, just display as "R = <value>"
    return f"$R$ = {x:.2f}"

from fractions import Fraction

def format_fraction(x):
    frac = Fraction(x).limit_denominator(1000)
    return f"{frac.numerator}" if frac.denominator == 1 else f"{frac.numerator}/{frac.denominator}"


# -------------------------------
# Load and process CSV files

# Load enkf CSV.
enkf_df = pd.read_csv('cache/enkf_1_4_cov.csv')
enkf_df = enkf_df.rename(columns={
    'measurement_covariance': 'covariance',
    'update_method': 'method'
})
if 'bandwidth' not in enkf_df.columns:
    enkf_df['bandwidth'] = 1.0
if 'random' not in enkf_df.columns:
    enkf_df['random'] = enkf_df.index.astype(str)
# Process the "rmse" column (string representation of a list) into a scalar (its mean).
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
gmf_df = gmf_df.rename(columns={'sampling': 'method'})
gmf_df['covariance'] = gmf_df['covariance'].astype(float)
gmf_df['bandwidth'] = gmf_df['bandwidth'].astype(float)
# (Assume gmf_df already has a scalar RMSE column.)

# -------------------------------
# Combine dataframes and filter out BRUEnKF rows.
df_all = pd.concat([enkf_df, gmf_df], ignore_index=True)
df_all = df_all[df_all['method'] != 'bruenkf_update']

# Map raw method names to our three desired labels.
method_map = {
    "enkf_update": "EnKF",
    "ikeda_rejection_sample_batch": "DI-EnGMF",
    "sample_gaussian_mixture": "EnGMF"
}
df_all['sampling_label'] = df_all['method'].map(method_map)

# -------------------------------
# Expand EnKF rows so that they appear for every unique bandwidth.
all_bw = np.sort(df_all['bandwidth'].unique())  # e.g. [0.333333, 0.666667, 1.0]
enkf_rows = df_all[df_all['sampling_label'] == "EnKF"].copy()
expanded_list = []
for bw in all_bw:
    tmp = enkf_rows.copy()
    tmp['bandwidth'] = bw
    expanded_list.append(tmp)
enkf_expanded = pd.concat(expanded_list, ignore_index=True)
# Remove original EnKF rows and add the expanded ones.
df_non_enkf = df_all[df_all['sampling_label'] != "EnKF"]
df_all = pd.concat([df_non_enkf, enkf_expanded], ignore_index=True)

# -------------------------------
# Create a facet label column for covariance.
df_all['cov_label'] = df_all['covariance'].apply(format_cov_label)

# -------------------------------
# Aggregate RMSE statistics over Monte Carlo runs.
group_cols = ["bandwidth", "covariance", "ensemble_size", "sampling_label"]
df_stats = (
    df_all.groupby(group_cols)["RMSE"]
    .agg(mean_rmse="mean", std_rmse="std")
    .reset_index()
)
df_stats['cov_label'] = df_stats['covariance'].apply(format_cov_label)

# -------------------------------
# Custom facet plotting function.
# This function plots error bars for each method in the facet.
def facet_plot(data, **kwargs):
    ax = plt.gca()
    style_map = {
        "EnKF": {"marker": "o", "linestyle": "-"},
        "DI-EnGMF": {"marker": "s", "linestyle": ":"},
        "EnGMF": {"marker": "^", "linestyle": "-."}
    }
    for method, group in data.groupby("sampling_label"):
        ax.errorbar(
            group["ensemble_size"],
            group["mean_rmse"],
            yerr=group["std_rmse"],
            marker=style_map[method]["marker"],
            linestyle=style_map[method]["linestyle"],
            lw=6,
            ms=10,
            capsize=4,
            label=method
        )
    ax.set_xlabel("Ensemble Size")
    ax.set_ylabel("RMSE")

# -------------------------------
# Plotting: One figure per unique bandwidth.
sns.set(context="talk", style="whitegrid", font_scale=2.3)
palette = sns.color_palette("colorblind", n_colors=3)
os.makedirs("plots", exist_ok=True)
unique_bw = np.sort(df_stats['bandwidth'].unique())
#sns.set(font_scale=1)
#sns.set_context("notebook", font_scale=5)
for bw in unique_bw:
    df_bw = df_stats[df_stats['bandwidth'] == bw]
    # Create a FacetGrid with row facets by covariance.
    g = sns.FacetGrid(
        data=df_bw,
        row="cov_label",       # facets: one row per covariance value
        sharey=True,
        height=5,
        aspect=2,
        palette=palette,
        legend_out=False,
    )
    g.map_dataframe(facet_plot)
    #g.fig.subplots_adjust(top=0.9, hspace=0.1, wspace=0.2)
    g.add_legend(ncols=3, loc='lower center', bbox_to_anchor=(0.5, -3.655))
    # sns.move_legend(g, loc='center right', bbox_to_anchor=(1.2, 0.5))
    g.set_axis_labels("Ensemble Size", "RMSE")
    # Remove the default row titles.
    for ax in g.axes.flat:
        ax.set_title("")
    # Now add custom text on the right side for each facet.
    for i, ax in enumerate(g.axes.flat):
        # Get the facet value from the row variable.
        cov_label = g.row_names[i]
        ax.text(1.02, 0.5, cov_label, transform=ax.transAxes,
                va="center", ha="left", fontweight="bold", rotation=270)
    g.fig.suptitle(f"$s_\\beta$ = {format_fraction(bw)}", weight='bold', y=1, x=0.5)
    # for ax in g.axes.flat:
        # ax.set_xticks(ax.get_xticks())
    # for ax in g.axes:
        # plt.setp(ax.get_xticklabels(), visible=True, rotation=45)

    fig_filename = f"plots/rmse_plot_bw_{bw:.2f}.pdf"
    g.fig.savefig(fig_filename, format="pdf", bbox_inches="tight")
    plt.show()
