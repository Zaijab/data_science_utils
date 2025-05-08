import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional, Union
import seaborn as sns
from pathlib import Path


def plot_rmse_comparison(
    ensemble_sizes: List[int],
    etpf_rmses: List[float],
    enkf_rmses: List[float],
    title: str = "RMSE vs Ensemble Size",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Union[str, Path, bool]] = None,
    style: str = "seaborn-v0_8-whitegrid",
    show_plot: bool = True,
    log_scale: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the RMSE comparison between ETPF and EnKF filters against ensemble size.

    Parameters
    ----------
    ensemble_sizes : List[int]
        List of ensemble sizes used in the experiment.
    etpf_rmses : List[float]
        List of RMSE values for the ETPF filter.
    enkf_rmses : List[float]
        List of RMSE values for the EnKF filter.
    title : str, optional
        Plot title, by default "RMSE vs Ensemble Size".
    figsize : Tuple[int, int], optional
        Figure size (width, height) in inches, by default (10, 6).
    save_path : Optional[Union[str, Path]], optional
        Path to save the figure, by default None.
    style : str, optional
        Matplotlib style to use, by default "seaborn-v0_8-whitegrid".
    show_plot : bool, optional
        Whether to display the plot, by default True.
    log_scale : bool, optional
        Whether to use logarithmic scale for the y-axis, by default False.
    add_trend : bool, optional
        Whether to add trend lines, by default True.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The figure and axes objects for further customization if needed.

    Examples
    --------
    >>> ensemble_sizes = list(range(10, 200, 10))
    >>> fig, ax = plot_rmse_comparison(ensemble_sizes, etpf_rmses, enkf_rmses)
    """
    # Validate inputs
    if len(ensemble_sizes) != len(etpf_rmses) or len(ensemble_sizes) != len(enkf_rmses):
        raise ValueError("All input lists must have the same length")

    # Set the style
    with plt.style.context(style):
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # Plot the data
        ax.plot(
            ensemble_sizes,
            etpf_rmses,
            "o--",
            color="#1f77b4",
            label="ETPF Genetic",
            linewidth=2,
            markersize=8,
        )
        ax.plot(
            ensemble_sizes,
            enkf_rmses,
            "s-",
            color="#ff7f0e",
            label="ETPF Sinkhorn",
            linewidth=2,
            markersize=8,
        )

        # No trend lines as requested

        # Set log scale if requested
        if log_scale:
            ax.set_yscale("log")

        # Add grid, labels, and title
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Ensemble Size", fontsize=12)
        ax.set_ylabel("RMSE", fontsize=12)
        ax.set_title(title, fontsize=14, pad=20)

        # Set x-axis ticks at intervals of 20
        x_ticks = np.arange(min(ensemble_sizes), max(ensemble_sizes) + 1, 20)
        ax.set_xticks(x_ticks)

        # Add legend
        ax.legend(fontsize=12, frameon=True, facecolor="white", edgecolor="#dddddd")

        # No annotations for minimum values as requested

        # Tight layout
        plt.tight_layout()

        # Save figure if path is provided
        if save_path:
            # If title is provided and save_path is not a complete path, use PROJECT_HEAD structure
            if isinstance(save_path, str) and not (
                "/" in save_path or "\\" in save_path
            ):
                import os

                # Create the plots directory if it doesn't exist
                os.makedirs(
                    os.path.join(os.environ.get("PROJECT_HEAD", "."), "plots"),
                    exist_ok=True,
                )
                # Use the title for filename if no specific name is given
                filename = (
                    title.replace(" ", "_").lower() + ".png"
                    if save_path is True
                    else save_path
                )
                full_path = os.path.join(
                    os.environ.get("PROJECT_HEAD", "."), "plots", filename
                )
                plt.savefig(full_path, dpi=300, bbox_inches="tight")
            else:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")

        # Show plot if requested
        if show_plot:
            plt.show()

        return fig, ax


# Example usage:
if __name__ == "__main__":
    # Sample data (from your code output)
    ensemble_sizes = list(range(10, 120, 10))
    etpf_genetic = [
        0.7417990241341957,
        0.7418011682742582,
        0.7418083052599671,
        0.7418072934905068,
        0.741805618727445,
        0.7417988611993223,
        0.741803556365289,
        0.7418018813817245,
        0.7418037258097507,
        0.7417993082778247,
        0.741802127790146,
    ]
    etpf_sinkhorn = [
        0.5201044997678486,
        0.4714536405634747,
        0.5111001978835521,
        0.5065053912963067,
        0.4748012186245475,
        0.4898295630066141,
        0.5102332482268868,
        0.3891214885826858,
        0.4598431649366838,
        0.4150996807930913,
        0.45437768389176164,
    ]

    # Plot the data
    fig, ax = plot_rmse_comparison(
        ensemble_sizes,
        etpf_genetic,
        etpf_sinkhorn,
        title="RMSE vs Ensemble Size for ETPF Filters with Different Solvers",
        save_path="rmse_comparison_genetic.png",
    )
