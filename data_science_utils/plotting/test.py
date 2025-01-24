import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('small_output.csv')

def plot_rmse_grid(df):
    """
    Takes a dataframe with columns:
      ["ensemble_size", "random", "covariance", "bandwidth", "sampling", "RMSE"]
    and produces a grid of subplots:
      Rows indexed by bandwidth,
      Columns indexed by measurement covariance,
      X-axis = ensemble_size, Y-axis = RMSE,
      Two lines in each plot (one per sampling method),
      Error bars computed across all random-seed runs (std dev or confidence interval).
    """

    # 1) Aggregate by (bandwidth, covariance, ensemble_size, sampling),
    #    compute mean RMSE and std dev across the random seeds.
    grouped = df.groupby(["bandwidth", "covariance", "ensemble_size", "sampling"])["RMSE"]
    df_stats = grouped.agg(mean_rmse=("mean"), std_rmse=("std")).reset_index()

    # 2) Set up a FacetGrid with row="bandwidth", col="covariance", hue="sampling".
    #    We'll plot mean_rmse vs ensemble_size with error bars = std_rmse.
    g = sns.FacetGrid(
        data=df_stats,
        row="bandwidth",
        col="covariance",
        hue="sampling",
        margin_titles=True,
        sharey=False
    )

    # 3) For each Facet, we draw error bars.
    #    x=ensemble_size, y=mean_rmse, yerr=std_rmse.
    g.map(
        plt.errorbar,
        "ensemble_size", "mean_rmse", "std_rmse",
        marker="o", capsize=3, linestyle="--"
    )

    # 4) Add legend.
    g.add_legend()

    # 5) Improve layout.
    g.set_axis_labels("Ensemble Size", "RMSE")
    g.set_titles(row_template="Bandwidth = {row_name}", col_template="Cov = {col_name}")
    plt.tight_layout()
    plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_rmse_grid(df):
    """
    Takes a DataFrame with columns:
      ["ensemble_size", "random", "covariance", "bandwidth", "sampling", "RMSE"]
    and produces a grid of subplots:
      - Rows: unique 'bandwidth'
      - Columns: unique 'covariance'
      - X-axis = 'ensemble_size'
      - Y-axis = mean of 'RMSE'
      - Error bars (std dev) across different random keys.
      - Two lines in each subplot (one for each sampling method).
    Legend is placed to the right outside of the plots.
    """

    # Map the sampling method to more readable labels
    sampler_map = {
        "sample_multivariate_normal": "Classical EnGMF",
        "rejection_sample_batch": "Discriminator EnGMF",
    }
    df["sampling_label"] = df["sampling"].map(sampler_map)

    # Group by (bandwidth, covariance, ensemble_size, sampling_label)
    grouped = df.groupby(["bandwidth", "covariance", "ensemble_size", "sampling_label"])["RMSE"]
    df_stats = grouped.agg(mean_rmse=("mean"), std_rmse=("std")).reset_index()

    # Choose a color-blind-friendly style and palette
    sns.set(context="talk", style="whitegrid", font_scale=1.2)
    palette = sns.color_palette("colorblind", n_colors=df_stats["sampling_label"].nunique())

    # Create a FacetGrid
    g = sns.FacetGrid(
        data=df_stats,
        row="bandwidth",
        col="covariance",
        hue="sampling_label",
        margin_titles=True,
        sharey=False,
        palette=palette,
        height=4,  # adjust height if desired
        aspect=1.3
    )

    # Plot mean RMSE vs. ensemble size with error bars = std dev
    g.map(
        plt.errorbar,
        "ensemble_size", "mean_rmse", "std_rmse",
        marker="o", capsize=3, linestyle="--"
    )

    # Move the legend to the right outside the grid
    g.add_legend(title="Sampling Method")
    g.fig.subplots_adjust(right=0.82)  # extra space on the right
    if g._legend:
        g._legend.set_bbox_to_anchor((1.02, 0.5))
        g._legend.set_loc("center left")

    # Set labels and titles
    g.set_axis_labels("Ensemble Size", "RMSE")
    g.set_titles(row_template="Bandwidth = {row_name}", col_template="Cov = {col_name}")

    plt.tight_layout()


# Usage (assuming your dataframe is "df"):
# plot_rmse_grid(df)

def plot_rmse_grid(df, filename="rmse_plot.svg"):
    """
    Takes a DataFrame with columns:
      ["ensemble_size", "random", "covariance", "bandwidth", "sampling", "RMSE"]
    and produces a grid of subplots:
      - Rows: unique 'bandwidth'
      - Columns: unique 'covariance'
      - X-axis: 'ensemble_size'
      - Y-axis: mean of 'RMSE'
      - Error bars (std dev) across different random keys.
      - Two lines in each subplot (one for each sampling method).
    Legend is placed to the right outside of the plots.

    Exports the figure as an SVG file with LaTeX rendering.
    """
    # Let Matplotlib use LaTeX for text rendering
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    # Increase figure DPI for better quality
    plt.rcParams["figure.dpi"] = 300

    # Map the sampling method to more readable labels
    sampler_map = {
        "sample_multivariate_normal": "Classical EnGMF",
        "rejection_sample_batch": "Discriminator EnGMF",
    }
    df["sampling_label"] = df["sampling"].map(sampler_map)

    # Group by (bandwidth, covariance, ensemble_size, sampling_label)
    grouped = df.groupby(["bandwidth", "covariance", "ensemble_size", "sampling_label"])["RMSE"]
    df_stats = grouped.agg(mean_rmse=("mean"), std_rmse=("std")).reset_index()

    # Choose a color-blind-friendly style and palette
    sns.set(context="talk", style="whitegrid", font_scale=1.2)
    palette = sns.color_palette("colorblind", n_colors=df_stats["sampling_label"].nunique())

    # Create a FacetGrid
    g = sns.FacetGrid(
        data=df_stats,
        row="bandwidth",
        col="covariance",
        hue="sampling_label",
        margin_titles=True,
        sharey=False,
        palette=palette,
        height=4,  # adjust as you wish
        aspect=1.3
    )

    # Plot mean RMSE vs. ensemble size with error bars = std dev
    g.map(
        plt.errorbar,
        "ensemble_size", "mean_rmse", "std_rmse",
        marker="o", capsize=3, linestyle="--"
    )

    # Manually place the legend on the right
    g.add_legend(title="Sampling Method")
    # Expand figure area on the right to accommodate the legend
    g.fig.subplots_adjust(right=0.8) 
    if g._legend:
        g._legend.set_bbox_to_anchor((0.95, 0.5))
        g._legend.set_loc("center left")

    # Set labels and titles
    g.set_axis_labels(r"\textbf{Ensemble Size}", r"\textbf{RMSE}")
    g.set_titles(
        row_template=r"Bandwidth ($\beta$): {row_name}", 
        col_template=r"Covariance ($R$): {col_name}"
    )

    # If you still find that the legend is getting cut off, try:
    #   for ax in g.axes.flatten():
    #       ax.legend(loc="best")
    # or
    g._legend.set_in_layout(True)
    plt.tight_layout()

    # Save the figure as an SVG with bbox_inches="tight" to ensure the legend is included
    g.fig.savefig(filename, format="svg", bbox_inches='tight')

    # Show if you want an interactive display
    plt.show()

# Example usage (assuming your final DataFrame is called df):
plot_rmse_grid(df)


