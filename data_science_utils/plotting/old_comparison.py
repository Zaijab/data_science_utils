import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm

from data_science_utils.dynamical_systems import flow, generate
from data_science_utils.filtering import ensemble_gaussian_mixture_filter_update_ensemble
from data_science_utils.measurement_functions import Distance

def filter_rmse_over_ensemble_sizes(
    ensemble_sizes,
    burn_in_time=500,
):
    """
    Bundles the given code into a single function and plots a time series
    over ensemble size of the RMSE.
    """
    measurement_time = 10 * burn_in_time

    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_disable_jit", False)

    key = jax.random.key(100)
    key, subkey = jax.random.split(key)
    measurement_device = Distance(jnp.array([[1.0]]))
    true_state = jnp.array([[1.25, 0.0]])

    rmses = []

    for ens_size in ensemble_sizes:
        # Compute Silverman's rule-of-thumb bandwidth factor
        silverman_bandwidth = (4 / (ens_size * (2 + 2))) ** (2 / (2 + 4))

        # Initialize filter ensemble
        key, subkey = jax.random.split(key)
        filter_ensemble = jax.random.multivariate_normal(
            key=subkey,
            shape=(ens_size,),
            mean=true_state,
            cov=jnp.eye(2),
        )

        # Define partial update function
        filter_update = partial(
            ensemble_gaussian_mixture_filter_update_ensemble,
            bandwidth_factor=silverman_bandwidth,
            measurement_device=measurement_device,
        )

        # Burn-in + measurement loop
        states = []
        covariances = []

        for t in tqdm(range(burn_in_time + measurement_time), leave=False):
            key, subkey = jax.random.split(key)
            filter_ensemble = filter_update(
                state=filter_ensemble,
                key=subkey,
                measurement=measurement_device(true_state),
            )
            if t >= burn_in_time:
                # Collect state deviations and ensemble covariances
                states.append(true_state - jnp.mean(filter_ensemble, axis=0))
                cov = jnp.cov(filter_ensemble.T)
                covariances.append(cov)

            # Flow dynamics
            filter_ensemble = flow(filter_ensemble)
            true_state = flow(true_state)

        # Compute RMSE
        states_array = jnp.array(states)
        e = jnp.expand_dims(states_array, -1)
        P = jnp.expand_dims(jnp.array(covariances), 1)
        rmse = jnp.sqrt(jnp.mean(e * e))
        print('hehehaha', rmse)
        rmses.append(rmse)

    # Plot RMSE vs ensemble size
    plt.figure(figsize=(6, 4))
    plt.plot(ensemble_sizes, rmses, marker='o')
    plt.xlabel('Ensemble Size')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Ensemble Size')
    plt.grid(True)
    plt.show()

#filter_rmse_over_ensemble_sizes(range(3, 120, 3))
# filter_rmse_over_ensemble_sizes([10])

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm

from data_science_utils.dynamical_systems import flow
from data_science_utils.filtering import ensemble_gaussian_mixture_filter_update_ensemble
from data_science_utils.measurement_functions import Distance

def monte_carlo_filter_rmse(
    ensemble_sizes,
    burn_in_time=500,
    mc_runs=20,
    seed=100,
):
    """
    Monte Carlo simulation of a single filter (ensemble_gaussian_mixture_filter_update_ensemble).
    For each ensemble size, repeats mc_runs times, and plots the time series of RMSE
    with 95% confidence intervals over those runs.
    """
    # Total time steps = burn_in_time + measurement_time
    measurement_time = 10 * burn_in_time
    total_time = burn_in_time + measurement_time

    # Set JAX configs
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_disable_jit", False)

    # Create a top-level PRNG key
    master_key = jax.random.key(seed)
    measurement_device = Distance(jnp.array([[1.0]]))

    # Prepare figure
    plt.figure(figsize=(7, 4.5))
    cmap = plt.get_cmap('tab10', len(ensemble_sizes))

    for idx, ens_size in enumerate(ensemble_sizes):
        # Array to store RMSE(t) for each run, shape = (mc_runs, measurement_time)
        all_rmse_time = []

        for run in range(mc_runs):
            # Split for each run so it starts with a fresh subkey
            master_key, subkey_init = jax.random.split(master_key)

            # True state reset
            true_state = jnp.array([[1.25, 0.0]])

            # Initialize filter ensemble
            silverman_bandwidth = (4 / (ens_size * (2 + 2))) ** (2 / (2 + 4))
            filter_ensemble = jax.random.multivariate_normal(
                key=subkey_init,
                shape=(ens_size,),
                mean=true_state,
                cov=jnp.eye(2),
            )

            # Define partial update function
            filter_update = partial(
                ensemble_gaussian_mixture_filter_update_ensemble,
                bandwidth_factor=silverman_bandwidth,
                measurement_device=measurement_device,
            )

            # We'll track instantaneous RMSE after burn-in
            rmse_time = []

            # Main loop
            for t in range(total_time):
                # Split key for each time step
                master_key, subkey_step = jax.random.split(master_key)

                # Filter update
                filter_ensemble = filter_update(
                    state=filter_ensemble,
                    key=subkey_step,
                    measurement=measurement_device(true_state),
                )

                # Collect stats after burn-in
                if t >= burn_in_time:
                    # Compute instantaneous RMSE
                    mean_est = jnp.mean(filter_ensemble, axis=0)  # [2,]
                    error = true_state - mean_est
                    rmse_t = jnp.sqrt(jnp.mean(error ** 2))
                    rmse_time.append(rmse_t)

                # Flow dynamics
                filter_ensemble = flow(filter_ensemble)
                true_state = flow(true_state)

            # Convert to an array (length = measurement_time)
            rmse_time = jnp.array(rmse_time)
            all_rmse_time.append(rmse_time)

        # Stack across runs -> shape = (mc_runs, measurement_time)
        all_rmse_time = jnp.stack(all_rmse_time, axis=0)

        # Compute mean and 2.5/97.5 quantiles across runs at each time
        mean_rmse = jnp.mean(all_rmse_time, axis=0)
        lower_ci = jnp.percentile(all_rmse_time, 2.5, axis=0)
        upper_ci = jnp.percentile(all_rmse_time, 97.5, axis=0)

        # Time axis for measurement window
        time_axis = jnp.arange(measurement_time)

        # Plot
        color = cmap(idx)
        plt.plot(time_axis, mean_rmse, color=color, label=f'Ensemble={ens_size}')
        plt.fill_between(time_axis, lower_ci, upper_ci, color=color, alpha=0.2)

    plt.xlabel("Time (after burn-in)")
    plt.ylabel("RMSE")
    plt.title("RMSE over Time with 95% CI (Monte Carlo)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage:
# monte_carlo_filter_rmse([10, 20], burn_in_time=500, mc_runs=20)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm

# These imports assume your package structure as described.
# Adjust paths if needed.
from data_science_utils.dynamical_systems import flow
from data_science_utils.filtering import ensemble_gaussian_mixture_filter_update_ensemble
from data_science_utils.measurement_functions import Distance

def monte_carlo_filter_rmse_by_ens_size(
    ensemble_sizes,
    burn_in_time=500,
    measurement_time_factor=10,
    mc_runs=20,
    seed=100
):
    """
    Runs Monte Carlo simulations for ensemble_gaussian_mixture_filter_update_ensemble
    for multiple ensemble sizes. Plots the mean RMSE (after burn-in) vs. ensemble size,
    along with 95% confidence intervals.

    Because you mentioned that jax.random.key is correct for your environment,
    this code uses jax.random.key(...) and subkeys from there. If your setup
    still triggers Key[] type-checking issues, ensure your library is expecting
    a uint32[2] array or adapt the usage to your local environment's convention.

    The final plot has ensemble size on the x-axis and the RMSE on the y-axis,
    with one point per ensemble size (and 95% CIs).
    """

    total_time = burn_in_time * (1 + measurement_time_factor)
    # We'll compute a single "summary RMSE" for each run by averaging the RMSE
    # over the measurement window [burn_in_time : burn_in_time+measurement_time].

    # Initialize the top-level key with your custom jax.random.key method.
    # If this fails for type-checking reasons, you may need jax.random.PRNGKey instead.
    master_key = jax.random.key(seed)

    measurement_device = Distance(jnp.array([[1.0]]))

    rmse_means = []
    rmse_lower = []
    rmse_upper = []

    for ens_size in ensemble_sizes:
        # Collect average RMSE over the measurement window for each run
        mc_results = []

        for _ in tqdm(range(mc_runs)):
            master_key, subkey_init = jax.random.split(master_key)

            # Reset the true state each run
            true_state = jnp.array([[1.25, 0.0]])

            # Compute Silverman's rule-of-thumb bandwidth factor
            silverman_bandwidth = (4 / (ens_size * (2 + 2))) ** (2 / (2 + 4))

            # Initialize filter ensemble around the true state
            filter_ensemble = jax.random.multivariate_normal(
                key=subkey_init,
                shape=(ens_size,),
                mean=true_state,
                cov=jnp.eye(2),
            )

            filter_update = partial(
                ensemble_gaussian_mixture_filter_update_ensemble,
                bandwidth_factor=silverman_bandwidth,
                measurement_device=measurement_device,
            )

            # We'll store RMSE(t) after burn-in
            rmse_over_time = []

            for t in range(total_time):
                master_key, subkey_step = jax.random.split(master_key)

                filter_ensemble = filter_update(
                    state=filter_ensemble,
                    key=subkey_step,
                    measurement=measurement_device(true_state),
                )

                if t >= burn_in_time:
                    # Compute instantaneous RMSE
                    mean_est = jnp.mean(filter_ensemble, axis=0)  # shape (2,)
                    error = true_state - mean_est  # shape (1,2)
                    rmse_t = jnp.sqrt(jnp.mean(error ** 2))
                    rmse_over_time.append(rmse_t)

                # Advance both the filter ensemble and the true state
                filter_ensemble = flow(filter_ensemble)
                true_state = flow(true_state)

            # Convert the list of RMSEs to a JAX array
            rmse_array = jnp.array(rmse_over_time)  # shape (measurement_time,)

            # Take the average over the measurement window
            # (Alternatively, you could take the final RMSE or median, etc.)
            mc_results.append(jnp.mean(rmse_array))

        # Convert results to a JAX array for percentile computations
        mc_results = jnp.array(mc_results)
        mean_val = jnp.mean(mc_results)
        lower_95 = jnp.percentile(mc_results, 2.5)
        upper_95 = jnp.percentile(mc_results, 97.5)

        rmse_means.append(mean_val)
        rmse_lower.append(lower_95)
        rmse_upper.append(upper_95)

    # Plot: X-axis = ensemble_sizes, Y-axis = average RMSE, with 95% CI
    plt.figure(figsize=(6, 4))
    ensemble_sizes = jnp.array(ensemble_sizes, dtype=float)
    rmse_means = jnp.array(rmse_means)
    rmse_lower = jnp.array(rmse_lower)
    rmse_upper = jnp.array(rmse_upper)

    # Plot the mean as points
    plt.plot(ensemble_sizes, rmse_means, color='tab:blue', marker='o', label='GM Filter')

    # Add error bars for 95% CI
    plt.fill_between(
        ensemble_sizes,
        rmse_lower,
        rmse_upper,
        color='tab:blue',
        alpha=0.2
    )

    plt.xlabel('Ensemble Size')
    plt.ylabel('RMSE')
    plt.title('Mean RMSE vs. Ensemble Size with 95% CI')
    plt.grid(True)
    plt.legend()
    plt.show()

# Example usage: 
# monte_carlo_filter_rmse_by_ens_size([10, 20], burn_in_time=500, mc_runs=20)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm

from data_science_utils.dynamical_systems import flow
from data_science_utils.filtering import (
    ensemble_gaussian_mixture_filter_update_ensemble,
    discriminator_ensemble_gaussian_mixture_filter_update_ensemble
)
from data_science_utils.measurement_functions import Distance

def monte_carlo_compare_two_filters(
    ensemble_sizes,
    burn_in_time=500,
    measurement_time_factor=10,
    mc_runs=20,
    seed=100,
):
    """
    Compares two filters:
      1) ensemble_gaussian_mixture_filter_update_ensemble
      2) discriminator_ensemble_gaussian_mixture_filter_update_ensemble

    For each ensemble size, we run multiple Monte Carlo simulations for each filter.
    We then plot the mean RMSE vs. ensemble size with 95% confidence intervals.
    """

    total_time = burn_in_time * (1 + measurement_time_factor)

    # Create a top-level key with your method: jax.random.key(...)
    master_key = jax.random.key(seed)
    measurement_device = Distance(jnp.array([[1.0]]))

    # Define the two filters we want to compare, with partials
    def gm_filter(bandwidth_factor):
        return partial(
            ensemble_gaussian_mixture_filter_update_ensemble,
            bandwidth_factor=bandwidth_factor,
            measurement_device=measurement_device
        )

    def disc_filter(bandwidth_factor):
        return partial(
            discriminator_ensemble_gaussian_mixture_filter_update_ensemble,
            bandwidth_factor=bandwidth_factor,
            measurement_device=measurement_device,
            ninverses=8
        )

    # We'll collect data for each filter in dict form
    filter_defs = {
        "GM Filter": gm_filter,
        "Disc Filter": disc_filter,
    }

    # Prepare to store results for each filter
    results = {
        "GM Filter": {"mean": [], "low": [], "high": []},
        "Disc Filter": {"mean": [], "low": [], "high": []},
    }

    # Loop over ensemble sizes
    for ens_size in tqdm(ensemble_sizes):

        # Compute Silverman's bandwidth
        silverman_bandwidth = (4 / (ens_size * (2 + 2))) ** (2 / (2 + 4))

        # For each filter, do mc_runs
        for filter_label, filter_maker in filter_defs.items():
            mc_vals = []

            for _ in range(mc_runs):
                master_key, subkey_init = jax.random.split(master_key)
                true_state = jnp.array([[1.25, 0.0]])

                filter_ensemble = jax.random.multivariate_normal(
                    key=subkey_init,
                    shape=(ens_size,),
                    mean=true_state,
                    cov=jnp.eye(2),
                )

                # Build the partial for this filter
                filter_update = filter_maker(silverman_bandwidth)

                # Collect RMSE for each time after burn-in
                rmse_over_time = []

                for t in range(total_time):
                    master_key, subkey_step = jax.random.split(master_key)

                    filter_ensemble = filter_update(
                        state=filter_ensemble,
                        key=subkey_step,
                        measurement=measurement_device(true_state),
                    )

                    if t >= burn_in_time:
                        mean_est = jnp.mean(filter_ensemble, axis=0)
                        err = true_state - mean_est
                        rmse_t = jnp.sqrt(jnp.mean(err ** 2))
                        rmse_over_time.append(rmse_t)

                    filter_ensemble = flow(filter_ensemble)
                    true_state = flow(true_state)

                # Average RMSE over measurement period
                mc_vals.append(jnp.mean(jnp.array(rmse_over_time)))

            mc_vals = jnp.array(mc_vals)
            mean_val = jnp.mean(mc_vals)
            low_95 = jnp.percentile(mc_vals, 2.5)
            high_95 = jnp.percentile(mc_vals, 97.5)

            results[filter_label]["mean"].append(mean_val)
            results[filter_label]["low"].append(low_95)
            results[filter_label]["high"].append(high_95)

    # Convert lists to arrays
    xvals = jnp.array(ensemble_sizes, dtype=float)
    for filt in filter_defs.keys():
        results[filt]["mean"] = jnp.array(results[filt]["mean"])
        results[filt]["low"] = jnp.array(results[filt]["low"])
        results[filt]["high"] = jnp.array(results[filt]["high"])

    # Plot
    plt.figure(figsize=(7, 4.5))
    colors = {"GM Filter": "tab:blue", "Disc Filter": "tab:orange"}

    for filt in filter_defs.keys():
        plt.plot(
            xvals,
            results[filt]["mean"],
            marker='o',
            label=filt,
            color=colors[filt],
        )
        plt.fill_between(
            xvals,
            results[filt]["low"],
            results[filt]["high"],
            color=colors[filt],
            alpha=0.2,
        )

    plt.xlabel("Ensemble Size")
    plt.ylabel("Mean RMSE (avg over measurement window)")
    plt.title("Monte Carlo Filter Comparison (95% CI)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage:
# monte_carlo_compare_two_filters([10, 20], burn_in_time=500, mc_runs=20)

import itertools
import jax
import jax.numpy as jnp
import pandas as pd
from tqdm import tqdm
from functools import partial

import matplotlib.pyplot as plt

# Adjust imports as needed for your local setup:
from data_science_utils.dynamical_systems import flow
from data_science_utils.filtering import (
    ensemble_gaussian_mixture_filter_update_ensemble,
    discriminator_ensemble_gaussian_mixture_filter_update_ensemble,
)
from data_science_utils.measurement_functions import Distance

def monte_carlo_experiments_with_R_and_bandwidth(
    R_values,
    bandwidth_values,
    ensemble_sizes,
    burn_in_time=500,
    measurement_time_factor=10,
    mc_runs=20,
    seed=100,
):
    """
    Runs Monte Carlo simulations for two filters:
        1) ensemble_gaussian_mixture_filter_update_ensemble
        2) discriminator_ensemble_gaussian_mixture_filter_update_ensemble
    across multiple settings of:
        - Measurement covariance R
        - Bandwidth factor (in this code, set by e.g. Silverman or user-defined)
        - Ensemble size
    and stores the results in a Pandas DataFrame.

    Each row in the resulting DataFrame provides:
        Filter (GM or Disc),
        R (measurement covariance),
        Bandwidth (factor),
        EnsembleSize,
        MeanRMSE,
        LowerCI (2.5% percentile),
        UpperCI (97.5% percentile).

    You can then plot or analyze these results at will.
    """

    total_time = burn_in_time * (1 + measurement_time_factor)

    # We will collect all experiment results into rows of a DataFrame
    results = []

    # Two filters we want to compare; each is a function returning
    # a partial(...) that configures the filter update.
    filter_defs = {
        "GM Filter": lambda bandwidth, measurement_device: partial(
            ensemble_gaussian_mixture_filter_update_ensemble,
            bandwidth_factor=bandwidth,
            measurement_device=measurement_device,
        ),
        "Disc Filter": lambda bandwidth, measurement_device: partial(
            discriminator_ensemble_gaussian_mixture_filter_update_ensemble,
            bandwidth_factor=bandwidth,
            measurement_device=measurement_device,
            ninverses=8,
        ),
    }

    # Master key for the entire experiment (assuming jax.random.key(...) is correct in your setup)
    master_key = jax.random.key(seed)

    # We'll iterate over all combinations of R, bandwidth, ens_size, and filter
    combos = list(
        itertools.product(
            R_values,
            bandwidth_values,
            ensemble_sizes,
            filter_defs.items()
        )
    )

    # TQDM progress bar over all combos
    with tqdm(total=len(combos)) as pbar:
        for (R, bandwidth, ens_size, (filt_label, filt_builder)) in combos:
            # Monte Carlo loop
            mc_vals = []

            # Build measurement device with this R
            measurement_device = Distance(jnp.array([[R]]))

            for _ in range(mc_runs):
                # Get a key for this run
                master_key, subkey_init = jax.random.split(master_key)

                # Reset the true state each run
                true_state = jnp.array([[1.25, 0.0]])

                # Draw initial ensemble
                filter_ensemble = jax.random.multivariate_normal(
                    key=subkey_init,
                    shape=(ens_size,),
                    mean=true_state,
                    cov=jnp.eye(2),
                )

                # Build the partial for this filter (bandwidth & measurement device)
                filter_update = filt_builder(bandwidth, measurement_device)

                # Track RMSE after burn-in
                rmse_over_time = []

                for t in range(total_time):
                    # Split key for each time step
                    master_key, subkey_step = jax.random.split(master_key)

                    # Filter update
                    filter_ensemble = filter_update(
                        state=filter_ensemble,
                        key=subkey_step,
                        measurement=measurement_device(true_state),
                    )

                    # Collect RMSE after burn-in
                    if t >= burn_in_time:
                        mean_est = jnp.mean(filter_ensemble, axis=0)
                        error = true_state - mean_est
                        rmse_t = jnp.sqrt(jnp.mean(error ** 2))
                        rmse_over_time.append(rmse_t)

                    # Advance dynamical system
                    filter_ensemble = flow(filter_ensemble)
                    true_state = flow(true_state)

                # Mean RMSE in the measurement window
                mc_vals.append(jnp.mean(jnp.array(rmse_over_time)))

            # Compute statistics: mean, 2.5% and 97.5% percentile
            mc_vals = jnp.array(mc_vals)
            mean_rmse = float(jnp.mean(mc_vals))
            lower_ci = float(jnp.percentile(mc_vals, 2.5))
            upper_ci = float(jnp.percentile(mc_vals, 97.5))

            # Store results in a dictionary
            row = {
                "Filter": filt_label,
                "R": R,
                "Bandwidth": bandwidth,
                "EnsembleSize": ens_size,
                "MeanRMSE": mean_rmse,
                "LowerCI": lower_ci,
                "UpperCI": upper_ci,
            }
            results.append(row)

            # Update progress bar
            pbar.update(1)

    # Convert to DataFrame
    df = pd.DataFrame(results)
    return df


if __name__ == "__main__":
    # Example usage:
    R_values = [0.1, 1.0]
    bandwidth_values = [0.3, 0.5]
    ensemble_sizes = [10, 20]

    # df_results = monte_carlo_experiments_with_R_and_bandwidth(
    #     R_values=R_values,
    #     bandwidth_values=bandwidth_values,
    #     ensemble_sizes=ensemble_sizes,
    #     burn_in_time=2,
    #     measurement_time_factor=5,
    #     mc_runs=5,
    #     seed=100,
    # )

    print("DataFrame of results:")
    print(df_results)

    # Example plotting snippet (customizable):
    # Here we just group by (Filter, R, Bandwidth, EnsembleSize).
    # The DataFrame already has the means and confidence intervals.    
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for filt_label, df_filt in df_results.groupby("Filter"):
        # We'll just do a basic example where we plot EnsembleSize vs. MeanRMSE
        # for each (R, Bandwidth) combination, on the same axis.
        # In practice, you might want subplots or facets.
        # We'll color by (R, Bandwidth) to show how they differ.
        for (rval, bval), df_sub in df_filt.groupby(["R", "Bandwidth"]):
            xvals = df_sub["EnsembleSize"].values
            yvals = df_sub["MeanRMSE"].values
            low_ci = df_sub["LowerCI"].values
            high_ci = df_sub["UpperCI"].values

            # Make a label that includes R and bandwidth
            label_str = f"{filt_label}, R={rval}, BW={bval}"

            ax.plot(xvals, yvals, marker='o', label=label_str)
            ax.fill_between(xvals, low_ci, high_ci, alpha=0.2)

    ax.set_xlabel("Ensemble Size")
    ax.set_ylabel("Mean RMSE")
    ax.set_title("Filter Comparison Across R & Bandwidth")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def plot_time_series_grid(df, R_values, bandwidth_values):
    """
    Creates a grid of time-series subplots:
      Rows: each R in R_values
      Columns: each bandwidth in bandwidth_values

    Within each subplot, we plot the RMSE vs. time for each (Filter, EnsembleSize).
    Shaded region for the MC 95% percentile across runs.
    """
    # Create subplots
    nrows = len(R_values)
    ncols = len(bandwidth_values)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.0 * ncols, 3.2 * nrows),
                            sharex=True, sharey=False)

    # Handle single-row or single-col edge cases gracefully
    if nrows == 1 and ncols == 1:
        axs = [[axs]]
    elif nrows == 1:
        axs = [axs]
    elif ncols == 1:
        axs = [[ax] for ax in axs]

    # Sort ensemble sizes so lines appear in ascending order
    df = df.sort_values("EnsembleSize")

    for i, R in enumerate(R_values):
        for j, bw in enumerate(bandwidth_values):
            ax = axs[i][j]

            # Subset the data for this R and bandwidth
            df_sub = df[(df["R"] == R) & (df["Bandwidth"] == bw)]

            # We'll plot lines for each combination of Filter + EnsembleSize
            for (filt_label, ens_size), df_grp in df_sub.groupby(["Filter", "EnsembleSize"]):
                # Group again by time to compute the mean/percentiles across MC runs
                # shape: (time -> multiple runs)
                grouped = df_grp.groupby("TimeIndex")["RMSE"]
                mean_rmse = grouped.mean()
                low_ci = grouped.quantile(0.025)
                high_ci = grouped.quantile(0.975)

                time_vals = mean_rmse.index.values
                ax.plot(
                    time_vals,
                    mean_rmse.values,
                    label=f"{filt_label}, N={ens_size}",
                    marker="o",
                    markersize=4
                )
                ax.fill_between(
                    time_vals,
                    low_ci.values,
                    high_ci.values,
                    alpha=0.2
                )

            ax.set_title(f"R={R}, BW={bw}")
            ax.set_xlabel("Time Step (after burn-in)")
            ax.set_ylabel("RMSE")
            ax.grid(True)

    # Put a legend in the lower-right subplot or an outside location
    # For multiple subplots, it's often clearer to do a single legend outside
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout()
    plt.show()

R_vals = [0.1, 1.0]
bw_vals = [0.3, 0.5]
ens_sizes = [10, 20]
plot_time_series_grid(df_results, R_vals, bw_vals)
