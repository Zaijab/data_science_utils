import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.concat([pd.read_csv('experiment.csv'), pd.read_csv('hpc_results.csv')])


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fractions import Fraction

def format_fraction(x):
    frac = Fraction(x).limit_denominator(1000)
    return f"{frac.numerator}" if frac.denominator == 1 else f"{frac.numerator}/{frac.denominator}"

def plot_rmse_grid(df, filename="rmse_plot.svg"):
    """
    Plots RMSE versus ensemble size with error bars computed across random-seed runs.
    The figure is faceted by bandwidth and measurement covariance (interpreted as $R^2$),
    with both parameters rendered as simple fractions.
    
    Parameters:
      df : DataFrame with columns
           ["ensemble_size", "random", "covariance", "bandwidth", "sampling", "RMSE"]
      filename : Name of the output SVG file.
    """
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["figure.dpi"] = 100

    sampler_map = {
        "sample_multivariate_normal": "Classical EnGMF",
        "rejection_sample_batch": "Discriminator EnGMF",
        "normalizing_flow": "Normalizing Flow EnGMF"
    }
    df["sampling_label"] = df["sampling"].map(sampler_map)

    # Aggregate RMSE statistics over random-seed runs.
    grouped = df.groupby(["bandwidth", "covariance", "ensemble_size", "sampling_label"])["RMSE"]
    df_stats = grouped.agg(mean_rmse=("mean"), std_rmse=("std")).reset_index()

    print(df_stats)

    # Create fraction labels for the facet parameters.
    df_stats["bandwidth_str"] = df_stats["bandwidth"].apply(format_fraction)
    df_stats["covariance_str"] = df_stats["covariance"].apply(format_fraction)

    sns.set(context="talk", style="whitegrid", font_scale=1.2)
    palette = sns.color_palette("colorblind", n_colors=df_stats["sampling_label"].nunique())

    # Set up the facet grid using the fraction-string labels.
    g = sns.FacetGrid(
        data=df_stats,
        row="bandwidth_str",
        col="covariance_str",
        hue="sampling_label",
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

    g.add_legend(title="Sampling Method")
    g.fig.subplots_adjust(right=0.8)
    if g._legend:
        g._legend.set_bbox_to_anchor((0.98, 0.5))
        g._legend.set_loc("center left")
    g.set_axis_labels(r"\textbf{Ensemble Size}", r"\textbf{RMSE}")
    g.set_titles(
        row_template=r"Bandwidth ($\beta$): {row_name}",
        col_template=r"Covariance ($R^2$): {col_name}"
    )
    g._legend.set_in_layout(True)
    plt.tight_layout()
    g.fig.savefig(filename, format="svg", bbox_inches="tight")
    plt.show()

    
# Example usage (assuming your final DataFrame is called df):
plot_rmse_grid(df)
