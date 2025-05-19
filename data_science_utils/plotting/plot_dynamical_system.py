from data_science_utils.dynamical_systems import AbstractDynamicalSystem

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Tuple, Union, List, Dict, Any, Callable
import numpy as np
from pathlib import Path


def plot_dynamical_system(
    system: AbstractDynamicalSystem,
    n_steps: int = 1000,
    n_trajectories: int = 10,
    seed: int = 42,
    save_path: Optional[str] = None,
    fig_size: Tuple[float, float] = (8, 6),
    show_transient: bool = False,
    transient_steps: int = 100,
    plot_type: str = "both",  # "phase", "time", or "both"
    plot_kwargs: Optional[Dict[str, Any]] = None,
) -> plt.Figure:
    """Create publication-quality plots for dynamical systems.

    Args:
        system: Dynamical system to plot
        n_steps: Number of time steps to simulate
        n_trajectories: Number of trajectories for stochastic systems
        seed: Random seed for reproducibility
        save_path: Path to save figure (None = don't save)
        fig_size: Figure size in inches
        show_transient: Whether to show transient behavior
        transient_steps: Number of initial steps considered transient
        plot_type: Type of plot ("phase", "time", or "both")
        plot_kwargs: Additional plotting parameters

    Returns:
        Matplotlib figure object
    """
    # Parameter validation and defaults
    plot_kwargs = plot_kwargs or {}
    dim = system.dimension

    # Set up publication-quality defaults
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.usetex": True,
            "axes.labelsize": 11,  # >= main text size
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 12,
            "figure.dpi": 300,
        }
    )

    # Wu-Okabe colorblind-safe palette (nature paper recommendation)
    cmap_cb = LinearSegmentedColormap.from_list(
        "Wu-Okabe",
        [
            "#000000",  # black
            "#E69F00",  # orange
            "#56B4E9",  # sky blue
            "#009E73",  # bluish green
            "#F0E442",  # yellow
            "#0072B2",  # blue
            "#D55E00",  # vermillion
            "#CC79A7",  # reddish purple
        ],
    )

    # Create figure with appropriate size
    if plot_type == "both" and dim > 1:
        fig = plt.figure(figsize=fig_size, constrained_layout=True)
        gs = GridSpec(2, 2, figure=fig, height_ratios=[2, 1])
        ax_phase = fig.add_subplot(gs[0, :])
        axes_time = [fig.add_subplot(gs[1, i]) for i in range(min(2, dim))]
    elif plot_type == "phase" and dim > 1:
        fig = plt.figure(figsize=fig_size, constrained_layout=True)
        ax_phase = fig.add_subplot(111)
        axes_time = []
    elif plot_type == "time":
        fig = plt.figure(figsize=fig_size, constrained_layout=True)
        axes_time = [fig.add_subplot(dim, 1, i + 1) for i in range(dim)]
        ax_phase = None
    else:
        fig = plt.figure(figsize=fig_size, constrained_layout=True)
        ax_phase = None
        axes_time = [fig.add_subplot(111)]

    # Generate state trajectories
    key = jax.random.key(seed)
    keys = jax.random.split(key, n_trajectories)

    all_states = []
    for k in keys:
        state = system.initial_state[None, ...]  # Add batch dimension
        states = [state]

        # Forward simulation
        for _ in range(n_steps):
            state = system.forward(state)
            states.append(state)

        trajectory = jnp.concatenate(states, axis=0)
        print(trajectory)
        all_states.append(trajectory)

    all_states = jnp.stack(all_states, axis=0)  # [n_traj, n_steps+1, dim]

    # Determine time range for plots
    time_range = jnp.arange(all_states.shape[1])
    steady_mask = (
        time_range >= transient_steps
        if not show_transient
        else jnp.ones_like(time_range, dtype=bool)
    )

    # Phase space plot (2D or 3D projection)
    if ax_phase is not None and dim > 1:
        if dim == 2:
            # 2D phase plot
            for traj_idx in range(n_trajectories):
                traj = all_states[traj_idx, steady_mask, :]
                ax_phase.plot(
                    traj[:, 0],
                    traj[:, 1],
                    linewidth=1.5,  # Thick enough for visibility
                    color=cmap_cb(traj_idx / max(1, n_trajectories - 1)),
                    alpha=0.8,
                )

            ax_phase.set_xlabel(r"$x_1$", fontsize=11)
            ax_phase.set_ylabel(r"$x_2$", fontsize=11)

        elif dim >= 3:
            # 3D phase plot (use first 3 dimensions)
            ax_phase = fig.add_subplot(gs[0, :], projection="3d")
            for traj_idx in range(n_trajectories):
                traj = all_states[traj_idx, steady_mask, :]
                ax_phase.plot(
                    traj[:, 0],
                    traj[:, 1],
                    traj[:, 2],
                    linewidth=1.5,
                    color=cmap_cb(traj_idx / max(1, n_trajectories - 1)),
                    alpha=0.8,
                )

            ax_phase.set_xlabel(r"$x_1$", fontsize=11)
            ax_phase.set_ylabel(r"$x_2$", fontsize=11)
            ax_phase.set_zlabel(r"$x_3$", fontsize=11)

        # Common phase plot settings
        ax_phase.set_title("Phase Space", fontsize=12)
        ax_phase.grid(True, linestyle="--", alpha=0.3)

        # Set equal aspect ratio for physical systems
        if plot_kwargs.get("equal_aspect", True):
            ax_phase.set_aspect("equal") if dim == 2 else None

    # Time series plots
    time_indices = jnp.arange(len(steady_mask))[steady_mask]
    for i, ax in enumerate(axes_time):
        if i >= dim:
            continue

        for traj_idx in range(n_trajectories):
            traj = all_states[traj_idx, :, :]
            ax.plot(
                time_indices,
                traj[steady_mask, i],
                linewidth=1.5,
                color=cmap_cb(traj_idx / max(1, n_trajectories - 1)),
                alpha=0.8,
            )

        ax.set_xlabel("Time Step", fontsize=11)
        ax.set_ylabel(f"$x_{i+1}$", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.3)

    # Add subplot labels (a), (b), etc.
    axes = [ax_phase] + axes_time if ax_phase is not None else axes_time
    for i, ax in enumerate(axes):
        if ax is not None:
            ax.text(
                -0.15,
                1.05,
                f"({chr(97+i)})",
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
            )

    # Set tight layout and save if path provided
    plt.tight_layout()

    if save_path:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig
